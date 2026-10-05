# SPDX-License-Identifier: GPL-3.0-or-later
#
# M3G (JSR-184 / Mobile 3D Graphics) importer for Blender 4.5
#
# Imports: scene graph (World/Group), meshes (triangle strips, all vertex array
# encodings), normals, UVs, vertex colors, materials, textures (Image2D),
# cameras, lights, animation (node transforms, camera FOV/clip, light color/
# intensity/spot angle, background crop/color), world background (color + image,
# screen-space), external files next to the .m3g, scrambled files.
# Not imported: skinning/morph targets (mesh imported as static base shape),
# fog, material/texture animation, sprites (empty), ambient light (empty).

bl_info = {
    "name": "M3G (JSR-184) Importer",
    "author": "Claude",
    "version": (1, 1, 0),
    "blender": (4, 5, 0),
    "location": "File > Import > M3G (.m3g)",
    "description": "Import Mobile 3D Graphics (.m3g) files",
    "category": "Import-Export",
}

import math
import os
import struct
import zlib

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector

M3G_MAGIC = b"\xabJSR184\xbb\r\n\x1a\n"

# Object type ids
T_APPEARANCE = 3
T_CAMERA = 5
T_COMPOSITING = 6
T_POLYGON_MODE = 8
T_GROUP = 9
T_IMAGE2D = 10
T_STRIP_ARRAY = 11
T_LIGHT = 12
T_MATERIAL = 13
T_MESH = 14
T_MORPHING_MESH = 15
T_SKINNED_MESH = 16
T_TEXTURE2D = 17
T_SPRITE = 18
T_VERTEX_ARRAY = 20
T_VERTEX_BUFFER = 21
T_WORLD = 22
T_CONTROLLER = 1
T_TRACK = 2
T_BACKGROUND = 4
T_KEYFRAMES = 19
T_EXTERNAL = 255

# Animation property ids
P_COLOR = 258
P_CROP = 259
P_FAR = 263
P_FOV = 264
P_INTENSITY = 265
P_NEAR = 267
P_ORIENTATION = 268
P_SCALE = 270
P_SPOT_ANGLE = 273
P_TRANSLATION = 275
TRS_PROPS = (P_ORIENTATION, P_SCALE, P_TRANSLATION)

# Keyframe interpolation -> Blender (SLERP/SQUAD approximated by linear + normalize)
INTERP = {176: 'LINEAR', 177: 'LINEAR', 178: 'BEZIER', 179: 'LINEAR', 180: 'CONSTANT'}

NODE_TYPES = {T_CAMERA, T_GROUP, T_LIGHT, T_MESH, T_MORPHING_MESH,
              T_SKINNED_MESH, T_SPRITE, T_WORLD}


class M3GError(Exception):
    pass


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------

class Reader:
    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0
        self.end = len(buf)

    def _need(self, n):
        if n < 0 or self.pos + n > self.end:
            raise M3GError("unexpected end of object data")

    def _take(self, fmt, size):
        self._need(size)
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def u8(self):
        return self._take("<B", 1)

    def u16(self):
        return self._take("<H", 2)

    def u32(self):
        return self._take("<I", 4)

    def i32(self):
        return self._take("<i", 4)

    def f32(self):
        return self._take("<f", 4)

    def boolean(self):
        return self.u8() != 0

    def skip(self, n):
        self._need(n)
        self.pos += n

    def raw(self, n):
        self._need(n)
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    def string(self):
        end = self.buf.find(b"\x00", self.pos, self.end)
        if end < 0:
            raise M3GError("unterminated string")
        v = self.buf[self.pos:end].decode("utf-8", "replace")
        self.pos = end + 1
        return v

    def floats(self, n):
        self._need(4 * n)
        v = struct.unpack_from("<%df" % n, self.buf, self.pos)
        self.pos += 4 * n
        return v

    def vec3(self):
        return self.floats(3)

    def matrix(self):
        return self.floats(16)

    def rgb(self):
        return (self.u8(), self.u8(), self.u8())

    def rgba(self):
        return (self.u8(), self.u8(), self.u8(), self.u8())

    def np_array(self, dtype, count):
        dt = np.dtype(dtype)
        if count == 0:
            return np.zeros(0, dtype=dt)
        size = dt.itemsize * count
        self._need(size)
        a = np.frombuffer(self.buf, dtype=dt, count=count, offset=self.pos)
        self.pos += size
        return a


# ---------------------------------------------------------------------------
# Object parsers
# ---------------------------------------------------------------------------

def _object3d(r, d=None):
    uid = r.u32()                 # userID
    n = r.u32()
    tracks = [r.u32() for _ in range(n)]
    params = []
    for _ in range(r.u32()):      # user parameters
        pid = r.u32()
        params.append((pid, r.raw(r.u32())))
    if d is not None:
        d["uid"] = uid
        d["tracks"] = tracks
        d["params"] = params


def param_text(val):
    try:
        return val.decode("utf-8").replace("\x00", "")
    except UnicodeDecodeError:
        return "hex:" + val.hex()


def _transformable(r, d):
    _object3d(r, d)
    d["T"] = (0.0, 0.0, 0.0)
    d["S"] = (1.0, 1.0, 1.0)
    d["R"] = None
    d["M"] = None
    if r.boolean():
        d["T"] = r.vec3()
        d["S"] = r.vec3()
        angle = r.f32()
        axis = r.vec3()
        d["R"] = (angle, axis)
    if r.boolean():
        d["M"] = r.matrix()


def _node(r, d):
    _transformable(r, d)
    d["render"] = r.boolean()
    r.boolean()                   # enablePicking
    d["alpha"] = r.u8()
    r.u32()                       # scope
    if r.boolean():               # alignment
        r.u8()
        r.u8()
        r.u32()
        r.u32()


def _h_external(r, d):
    d["uri"] = r.string()


def _h_sprite(r, d):
    _node(r, d)
    d["image"] = r.u32()
    d["appearance"] = r.u32()


def _h_group(r, d):
    _node(r, d)
    n = r.u32()
    d["children"] = [r.u32() for _ in range(n)]


def _h_world(r, d):
    _h_group(r, d)
    d["camera"] = r.u32()
    d["background"] = r.u32()


def _h_mesh(r, d):
    # Same prefix for Mesh / SkinnedMesh / MorphingMesh.
    _node(r, d)
    d["vb"] = r.u32()
    n = r.u32()
    subs = []
    for _ in range(n):
        ib = r.u32()
        ap = r.u32()
        subs.append((ib, ap))
    d["subs"] = subs


def _h_camera(r, d):
    _node(r, d)
    d["proj"] = r.u8()
    if d["proj"] == 48:           # generic
        d["pmatrix"] = r.matrix()
    else:
        d["fovy"] = r.f32()
        d["aspect"] = r.f32()
        d["near"] = r.f32()
        d["far"] = r.f32()


def _h_light(r, d):
    _node(r, d)
    d["att"] = (r.f32(), r.f32(), r.f32())
    d["color"] = r.rgb()
    d["mode"] = r.u8()
    d["intensity"] = r.f32()
    d["spot_angle"] = r.f32()
    d["spot_exp"] = r.f32()


def _h_appearance(r, d):
    _object3d(r, d)
    r.u8()                        # layer
    d["compositing"] = r.u32()
    d["fog"] = r.u32()
    d["polygon_mode"] = r.u32()
    d["material"] = r.u32()
    n = r.u32()
    d["textures"] = [r.u32() for _ in range(n)]


def _h_material(r, d):
    _object3d(r, d)
    r.rgb()                       # ambient
    d["diffuse"] = r.rgba()
    d["emissive"] = r.rgb()
    d["specular"] = r.rgb()
    d["shininess"] = r.f32()
    d["track"] = r.boolean()


def _h_polygon_mode(r, d):
    _object3d(r, d)
    d["culling"] = r.u8()
    d["shading"] = r.u8()
    d["winding"] = r.u8()
    r.boolean()
    r.boolean()
    r.boolean()


def _h_compositing(r, d):
    _object3d(r, d)
    r.boolean()
    r.boolean()
    r.boolean()
    r.boolean()
    d["blending"] = r.u8()
    d["alpha_threshold"] = r.u8()
    r.f32()
    r.f32()


def _h_texture2d(r, d):
    _transformable(r, d)
    d["image"] = r.u32()
    r.rgb()                       # blend color
    d["blending"] = r.u8()
    d["wrap_s"] = r.u8()
    d["wrap_t"] = r.u8()
    d["level_filter"] = r.u8()
    d["image_filter"] = r.u8()


def _h_image2d(r, d):
    _object3d(r, d)
    d["format"] = r.u8()
    d["mutable"] = r.boolean()
    d["width"] = r.u32()
    d["height"] = r.u32()
    d["palette"] = b""
    d["pixels"] = b""
    if not d["mutable"]:
        d["palette"] = r.raw(r.u32())
        d["pixels"] = r.raw(r.u32())


def _h_vertex_array(r, d):
    _object3d(r, d)
    csize = r.u8()
    ccount = r.u8()
    enc = r.u8()
    vcount = r.u16()
    if csize not in (1, 2) or not (1 <= ccount <= 4) or enc not in (0, 1):
        raise M3GError("bad vertex array header")
    raw = r.np_array("i1" if csize == 1 else "<i2", vcount * ccount)
    a = raw.reshape(vcount, ccount).astype(np.int64)
    if enc == 1:                  # delta encoding, wraps at component size
        a = np.cumsum(a, axis=0)
        if csize == 1:
            a = ((a + 128) & 0xFF) - 128
        else:
            a = ((a + 32768) & 0xFFFF) - 32768
    d["data"] = a
    d["csize"] = csize


def _h_vertex_buffer(r, d):
    _object3d(r, d)
    d["default_color"] = r.rgba()
    d["positions"] = r.u32()
    d["pbias"] = r.vec3()
    d["pscale"] = r.f32()
    d["normals"] = r.u32()
    d["colors"] = r.u32()
    n = r.u32()
    tcs = []
    for _ in range(n):
        idx = r.u32()
        bias = r.vec3()
        scale = r.f32()
        tcs.append((idx, bias, scale))
    d["texcoords"] = tcs


def _h_strip_array(r, d):
    _object3d(r, d)
    enc = r.u8()
    first = 0
    explicit = None
    if enc == 0:
        first = r.u32()
    elif enc == 1:
        first = r.u8()
    elif enc == 2:
        first = r.u16()
    elif enc in (128, 129, 130):
        n = r.u32()
        explicit = r.np_array({128: "<u4", 129: "u1", 130: "<u2"}[enc], n)
    else:
        raise M3GError("unknown index encoding %d" % enc)
    ns = r.u32()
    lengths = r.np_array("<u4", ns)
    if explicit is None:
        total = int(lengths.sum(dtype=np.int64))
        explicit = np.arange(first, first + total, dtype=np.int64)
    d["indices"] = explicit
    d["lengths"] = lengths


def _h_background(r, d):
    _object3d(r, d)
    d["color"] = r.rgba()
    d["image"] = r.u32()
    d["mode_x"] = r.u8()
    d["mode_y"] = r.u8()
    d["crop"] = (r.i32(), r.i32(), r.i32(), r.i32())
    d["depth_clear"] = r.boolean()
    d["color_clear"] = r.boolean()


def _h_controller(r, d):
    _object3d(r, d)
    d["speed"] = r.f32()
    d["weight"] = r.f32()
    r.i32()                       # active interval start (ignored)
    r.i32()                       # active interval end (ignored)
    d["refseq"] = r.f32()
    d["refworld"] = r.i32()


def _h_track(r, d):
    _object3d(r, d)
    d["seq"] = r.u32()
    d["ctrl"] = r.u32()
    d["prop"] = r.u32()


def _h_keyframes(r, d):
    _object3d(r, d)
    d["interp"] = r.u8()
    d["repeat"] = r.u8()
    enc = r.u8()
    d["duration"] = r.u32()
    d["first"] = r.u32()
    d["last"] = r.u32()
    cc = r.u32()
    kc = r.u32()
    if cc < 1 or cc > 64 or enc not in (0, 1, 2) or kc * 4 > r.end - r.pos:
        raise M3GError("bad keyframe sequence")
    times = np.empty(kc, dtype=np.int64)
    vals = np.empty((kc, cc), dtype=np.float64)
    if enc == 0:
        for i in range(kc):
            times[i] = r.u32()
            vals[i] = r.floats(cc)
    else:
        bias = np.array(r.floats(cc), dtype=np.float64)
        scale = np.array(r.floats(cc), dtype=np.float64)
        for i in range(kc):
            times[i] = r.u32()
            if enc == 1:
                vals[i] = bias + scale * (r.np_array("u1", cc) / 255.0)
            else:
                vals[i] = bias + scale * (r.np_array("<u2", cc) / 65535.0)
    d["times"] = times
    d["values"] = vals


HANDLERS = {
    T_APPEARANCE: _h_appearance,
    T_CAMERA: _h_camera,
    T_COMPOSITING: _h_compositing,
    T_POLYGON_MODE: _h_polygon_mode,
    T_GROUP: _h_group,
    T_IMAGE2D: _h_image2d,
    T_STRIP_ARRAY: _h_strip_array,
    T_LIGHT: _h_light,
    T_MATERIAL: _h_material,
    T_MESH: _h_mesh,
    T_MORPHING_MESH: _h_mesh,
    T_SKINNED_MESH: _h_mesh,
    T_TEXTURE2D: _h_texture2d,
    T_SPRITE: _h_sprite,
    T_VERTEX_ARRAY: _h_vertex_array,
    T_VERTEX_BUFFER: _h_vertex_buffer,
    T_WORLD: _h_world,
    255: _h_external,
    T_CONTROLLER: _h_controller,
    T_TRACK: _h_track,
    T_BACKGROUND: _h_background,
    T_KEYFRAMES: _h_keyframes,
}


# ---------------------------------------------------------------------------
# File container
# ---------------------------------------------------------------------------

def _section_chain(get, n):
    """Cheap structural check of a (virtual) byte stream get(i), length n:
    signature, then sections / uncompressed objects must chain exactly."""
    for i in range(12):
        if get(i) != M3G_MAGIC[i]:
            return False
    pos = 12
    while pos < n:
        if pos + 13 > n:
            return False
        comp = get(pos)
        total = get(pos + 1) | get(pos + 2) << 8 | get(pos + 3) << 16 | get(pos + 4) << 24
        if comp > 1 or total < 13 or pos + total > n:
            return False
        if comp == 0:
            p = pos + 9
            end = pos + total - 4
            while p < end:
                if p + 5 > end:
                    return False
                t = get(p)
                ln = get(p + 1) | get(p + 2) << 8 | get(p + 3) << 16 | get(p + 4) << 24
                if (t > 22 and t != 255) or p + 5 + ln > end:
                    return False
                p += 5 + ln
        pos += total
    return pos == n


def _checksums_ok(d):
    """Adler32 of every uncompressed section, zlib check for compressed ones."""
    if d[:12] != M3G_MAGIC:
        return False
    pos = 12
    n = len(d)
    while pos < n:
        if pos + 13 > n:
            return False
        comp = d[pos]
        total, ulen = struct.unpack_from("<II", d, pos + 1)
        if total < 13 or pos + total > n:
            return False
        if comp == 0:
            stored = struct.unpack_from("<I", d, pos + total - 4)[0]
            if (zlib.adler32(d[pos:pos + total - 4]) & 0xFFFFFFFF) != stored:
                return False
        else:
            try:
                if len(zlib.decompress(d[pos + 9:pos + total - 4])) != ulen:
                    return False
            except zlib.error:
                return False
        pos += total
    return True


M3G_SCRAMBLE_MAX = 8192


def deobfuscate(data):
    """Returns (data, note). Handles: plain files, junk before the signature,
    fully byte-reversed files, and files whose first/last K bytes were swapped
    and byte-reversed (K differs per file)."""
    n = len(data)
    if data[:12] == M3G_MAGIC:
        return data, None
    i = data.find(M3G_MAGIC, 0, 4096)
    if i > 0:
        return data[i:], "skipped %d bytes before the JSR184 signature" % i
    cand = data[::-1]
    if _checksums_ok(cand):
        return cand, "file was stored byte-reversed; restored"
    for K in range(1, min(n // 2, M3G_SCRAMBLE_MAX) + 1):
        def get(i, K=K):
            return data[n - 1 - i] if (i < K or i >= n - K) else data[i]
        if _section_chain(get, n):
            cand = data[n - K:][::-1] + data[K:n - K] + data[:K][::-1]
            if _checksums_ok(cand):
                return cand, "file head/tail was scrambled (%d bytes); restored" % K
    return data, None


class M3GFile:
    def __init__(self, data, path=None, shared=None, allow_external=True):
        self.objects = {}         # index -> (type, bytes)
        self.parsed = {}
        self.path = path
        self.allow_external = allow_external
        self._shared = shared if shared is not None else {"files": {}, "warnings": [], "notes": []}
        self.warnings = self._shared["warnings"]
        self.notes = self._shared["notes"]
        self._ext = {}            # external ref index -> M3GFile or None
        self._load(data)
        if path:
            self._shared["files"][os.path.normcase(os.path.abspath(path))] = self

    def _load(self, data):
        data, note = deobfuscate(data)
        if note:
            self.notes.append(note)
        if data[:12] != M3G_MAGIC:
            raise M3GError("Not an M3G file (no JSR184 signature, also tried "
                           "reversed and head/tail-scrambled layouts)")
        n = len(data)
        pos = 12
        index = 1                 # header object is index 1
        while pos < n:
            if n - pos < 13:
                break
            comp = data[pos]
            total, _ulen = struct.unpack_from("<II", data, pos + 1)
            if total < 13 or pos + total > n:
                self.warnings.append("File truncated or corrupt at byte %d" % pos)
                break
            body = data[pos + 9:pos + total - 4]
            if comp == 1:
                try:
                    body = zlib.decompress(body)
                except zlib.error:
                    try:
                        body = zlib.decompress(body, -15)
                    except zlib.error:
                        raise M3GError("Section at byte %d: zlib decompress failed" % pos)
            elif comp != 0:
                raise M3GError("Section at byte %d: unknown compression %d" % (pos, comp))
            pos += total

            p = 0
            bl = len(body)
            while p + 5 <= bl:
                otype = body[p]
                olen = struct.unpack_from("<I", body, p + 1)[0]
                if p + 5 + olen > bl:
                    raise M3GError("Object overruns its section")
                self.objects[index] = (otype, body[p + 5:p + 5 + olen])
                index += 1
                p += 5 + olen
        if not self.objects:
            raise M3GError("No objects found in file")

    def get(self, idx, *types):
        """Parsed object dict for index idx (0 = null), or None."""
        if not idx:
            return None
        d = self.parsed.get(idx)
        if d is None:
            ent = self.objects.get(idx)
            if ent is None:
                return None
            d = self._parse(idx, ent[0], ent[1])
            self.parsed[idx] = d
        if types and d["type"] not in types:
            return None
        return d

    def _parse(self, idx, otype, data):
        d = {"type": otype}
        fn = HANDLERS.get(otype)
        if fn is None:
            return d
        try:
            fn(Reader(data), d)
        except (M3GError, struct.error, ValueError) as e:
            self.warnings.append("Object %d (type %d) unreadable: %s" % (idx, otype, e))
            return {"type": -1}
        return d

    # ---- external references ------------------------------------------

    def _find_external(self, uri):
        if not self.path or "://" in uri:
            return None
        name = uri.split("#")[0].replace("\\", "/").lstrip("/")
        if not name:
            return None
        base = os.path.dirname(os.path.abspath(self.path))
        cand = os.path.normpath(os.path.join(base, name))
        try:
            inside = os.path.commonpath([base, cand]) == base
        except ValueError:
            inside = False
        if inside and os.path.isfile(cand):
            return cand
        bn = os.path.basename(name).lower()
        try:
            for fn in os.listdir(base):
                full = os.path.join(base, fn)
                if fn.lower() == bn and os.path.isfile(full):
                    return full
        except OSError:
            pass
        return None

    def _external(self, idx):
        """Loaded M3GFile for external reference object idx, or None."""
        if idx in self._ext:
            return self._ext[idx]
        if not self.allow_external:
            return None
        self._ext[idx] = None
        d = self.get(idx, T_EXTERNAL)
        if d is None:
            return None
        path = self._find_external(d["uri"])
        if path is None:
            self.warnings.append("External file not found next to the .m3g: %s" % d["uri"])
            return None
        key = os.path.normcase(os.path.abspath(path))
        other = self._shared["files"].get(key)
        if other is None:
            if len(self._shared["files"]) >= 64:
                self.warnings.append("Too many external files, skipped %s" % d["uri"])
                return None
            try:
                with open(path, "rb") as fh:
                    other = M3GFile(fh.read(), path=path, shared=self._shared)
            except (OSError, M3GError) as e:
                self.warnings.append("External file %s unreadable: %s" % (d["uri"], e))
                return None
        self._ext[idx] = other
        return other

    def resolve_image(self, idx):
        """(M3GFile, index) of the Image2D that object idx stands for."""
        if not idx or idx not in self.objects:
            return None
        t = self.objects[idx][0]
        if t == T_IMAGE2D:
            return (self, idx)
        if t != T_EXTERNAL:
            return None
        other = self._external(idx)
        if other is None:
            return None
        for i in sorted(other.objects):
            if other.objects[i][0] == T_IMAGE2D:
                return (other, i)
        self.warnings.append("External file %s contains no image" % self.get(idx)["uri"])
        return None

    def resolve_node(self, idx):
        """(M3GFile, index) of the first root scene node of an external file."""
        other = self._external(idx)
        if other is None:
            return None
        kids = set()
        for i, (t, _data) in other.objects.items():
            if t in (T_GROUP, T_WORLD):
                d = other.get(i)
                if d is not None and d["type"] in (T_GROUP, T_WORLD):
                    kids.update(d["children"])
        for i in sorted(other.objects):
            if other.objects[i][0] in NODE_TYPES and i not in kids:
                return (other, i)
        self.warnings.append("External file %s contains no scene node" % self.get(idx)["uri"])
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lin(v):
    v = v / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def rgb_lin(c):
    return (_lin(c[0]), _lin(c[1]), _lin(c[2]))


def lin_np(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def uv_name(k):
    return "UVMap" if k == 0 else "UVMap%d" % k


def strip_triangles(indices, lengths, cw, nverts):
    """Triangle strips -> (n,3) int64 array. Drops degenerate/out-of-range."""
    out = []
    pos = 0
    for L in lengths:
        L = int(L)
        s = np.asarray(indices[pos:pos + L], dtype=np.int64)
        pos += L
        if len(s) < 3:
            continue
        a = s[:-2]
        b = s[1:-1]
        c = s[2:]
        t = np.stack([a, b, c], axis=1)
        odd = (np.arange(len(a)) & 1).astype(bool)
        t[odd] = np.stack([b[odd], a[odd], c[odd]], axis=1)
        out.append(t)
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    t = np.concatenate(out)
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    ok &= (t.max(axis=1) < nverts) & (t.min(axis=1) >= 0)
    t = t[ok]
    if cw:
        t = t[:, [0, 2, 1]]
    return t


def is_identity16(m):
    return all(abs(m[i] - (1.0 if i % 5 == 0 else 0.0)) < 1e-6 for i in range(16))


def qmul(a, b):
    """Hamilton product of (n,4) quaternion arrays, (w, x, y, z)."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def socket_index(node, sock):
    for i, s in enumerate(node.inputs):
        if s.identifier == sock.identifier:
            return i
    raise ValueError("socket not found")


def assign_action(idb, action):
    ad = idb.animation_data
    if ad is None:
        ad = idb.animation_data_create()
    ad.action = action
    if hasattr(ad, "action_slot") and ad.action_slot is None:
        slots = getattr(action, "slots", None)
        if slots is not None and len(slots) > 0:
            try:
                ad.action_slot = slots[0]
            except Exception:
                pass


def find_loose_geometry(m3g):
    """Files without Mesh nodes (game assets built by app code): pair the
    unreferenced VertexBuffers, TriangleStripArrays and Appearances."""
    used_vb, used_ib, used_app = set(), set(), set()
    for idx, (t, _data) in m3g.objects.items():
        if t in (T_MESH, T_MORPHING_MESH, T_SKINNED_MESH):
            d = m3g.get(idx)
            if d is None or d["type"] != t:
                continue
            used_vb.add(d["vb"])
            for ib, ap in d["subs"]:
                used_ib.add(ib)
                used_app.add(ap)
        elif t == T_SPRITE:
            d = m3g.get(idx)
            if d is not None and d["type"] == t:
                used_app.add(d["appearance"])

    def pick(otype, used):
        return [i for i in sorted(m3g.objects)
                if m3g.objects[i][0] == otype and i not in used]

    vbs = pick(T_VERTEX_BUFFER, used_vb)
    ibs = pick(T_STRIP_ARRAY, used_ib)
    apps = pick(T_APPEARANCE, used_app)
    result = []
    for vi in vbs:
        vb = m3g.get(vi, T_VERTEX_BUFFER)
        pva = m3g.get(vb["positions"], T_VERTEX_ARRAY) if vb is not None else None
        if pva is None:
            continue
        nv = pva["data"].shape[0]
        fit = []
        for ii in ibs:
            ib = m3g.get(ii, T_STRIP_ARRAY)
            if ib is not None and len(ib["indices"]) and int(ib["indices"].max()) < nv:
                fit.append(ii)
        if not fit:
            continue
        ibs = [i for i in ibs if i not in fit]
        if len(apps) == len(fit):
            ap_list = list(apps)
        elif apps:
            ap_list = [apps[0]] * len(fit)
        else:
            ap_list = [0] * len(fit)
        result.append({"vb": vi, "subs": list(zip(fit, ap_list))})
    return result


def set_poly_prop(mesh, name, values):
    try:
        mesh.polygons.foreach_set(name, values)
    except Exception:
        for p, v in zip(mesh.polygons, values):
            setattr(p, name, v)


# ---------------------------------------------------------------------------
# Blender scene builder
# ---------------------------------------------------------------------------

class Importer:
    def __init__(self, m3g, context, z_up, scale, name, parent=None,
                 import_animation=True, import_background=True):
        self.m3g = m3g
        self.context = context
        self.z_up = z_up
        self.scale = scale
        self.name = name
        self.do_anim = import_animation
        self.do_bg = import_background
        # M3G is Y-up, camera looks down -Z. Blender is Z-up. C: (x,y,z)->(x,-z,y)
        self.C = Matrix.Rotation(math.radians(90.0), 4, 'X') if z_up else Matrix.Identity(4)
        self.Ci = self.C.inverted()
        sc = context.scene
        self.fps = sc.render.fps / sc.render.fps_base
        self.frame0 = sc.frame_start
        self.done = set()
        self.node_objs = {}
        self.mesh_cache = {}
        self.mat_cache = {}
        self.img_cache = {}
        self.warned = set()
        self.bg = None
        if parent is None:
            self.depth = 0
            self.coll = None
            self.created = []
            self.empties = []
            self.subs = []
            self.anim_state = {"last": 0.0}
        else:
            self.depth = parent.depth + 1
            self.coll = parent.coll
            self.created = parent.created
            self.empties = parent.empties
            self.subs = parent.subs
            self.anim_state = parent.anim_state

    def warn(self, msg):
        self.m3g.warnings.append(msg)

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            self.warn(msg)

    # ---- entry ----------------------------------------------------------

    def run(self):
        m3g = self.m3g
        children = set()
        for idx, (t, _data) in m3g.objects.items():
            if t in (T_GROUP, T_WORLD):
                d = m3g.get(idx)
                if d is not None and d["type"] in (T_GROUP, T_WORLD):
                    children.update(d["children"])
        roots = [i for i in sorted(m3g.objects)
                 if m3g.objects[i][0] in NODE_TYPES and i not in children]

        self.coll = bpy.data.collections.new(self.name)
        self.context.scene.collection.children.link(self.coll)
        for i in roots:
            self.build(i, None)
        for loose in find_loose_geometry(m3g):
            self.build_loose(loose)

        if not self.created:
            bpy.data.collections.remove(self.coll)
            raise M3GError("No scene nodes found in file")

        if self.do_bg:
            try:
                self.setup_background()
            except Exception as e:
                self.warn("Background import failed: %s" % e)
        if self.do_anim:
            for imp in [self] + list(self.subs):
                imp.animate_all()
            last = self.anim_state["last"]
            if last > 0.0:
                sc = self.context.scene
                sc.frame_end = max(sc.frame_end, int(math.ceil(last)))

        for idx in sorted(m3g.objects):
            if m3g.objects[idx][0] == 255 and idx not in m3g._ext:
                d = m3g.get(idx)
                if d is not None and d["type"] == 255:
                    self.warn("External reference not loaded: %s" % d["uri"])

        # size empties relative to scene extent so they are visible
        if self.empties:
            self.context.view_layer.update()
            pts = np.array([tuple(o.matrix_world.translation) for o in self.created])
            extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
            size = max(extent * 0.01, 0.25 * self.scale)
            for o in self.empties:
                o.empty_display_size = size

        # scene camera
        cam_obj = None
        for idx, obj in self.node_objs.items():
            d = m3g.get(idx)
            if d is not None and d["type"] == T_WORLD and d["camera"] in self.node_objs:
                cam_obj = self.node_objs[d["camera"]]
                break
        if cam_obj is None:
            for idx, obj in self.node_objs.items():
                if m3g.objects[idx][0] == T_CAMERA:
                    cam_obj = obj
                    break
        if cam_obj is not None:
            self.context.scene.camera = cam_obj

        for o in list(self.context.selected_objects):
            o.select_set(False)
        for o in self.created:
            o.select_set(True)
        self.context.view_layer.objects.active = self.created[0]
        return len(self.created)

    # ---- nodes ----------------------------------------------------------

    def node_matrix(self, d, camera_like=False, skip_m=False):
        T = Matrix.Translation(d["T"])
        R = Matrix.Identity(4)
        if d["R"] is not None:
            ang, ax = d["R"]
            v = Vector(ax)
            if v.length > 1e-12 and ang != 0.0:
                R = Matrix.Rotation(math.radians(ang), 4, v.normalized())
        S = Matrix.Diagonal((d["S"][0], d["S"][1], d["S"][2], 1.0))
        L = T @ R @ S
        if d["M"] is not None and not skip_m:
            m = d["M"]
            L = L @ Matrix((m[0:4], m[4:8], m[8:12], m[12:16]))
        if self.z_up:
            # cameras/lights point down local -Z in both systems -> C @ L
            L = (self.C @ L) if camera_like else (self.C @ L @ self.Ci)
        L.translation = L.translation * self.scale
        return L

    def m_conj(self, m):
        M = Matrix((m[0:4], m[4:8], m[8:12], m[12:16]))
        if self.z_up:
            M = self.C @ M @ self.Ci
        M.translation = M.translation * self.scale
        return M

    def has_trs_anim(self, d):
        for tid in d["tracks"]:
            tr = self.m3g.get(tid, T_TRACK)
            if tr is not None and tr["prop"] in TRS_PROPS:
                return True
        return False

    def build_loose(self, loose):
        name = "Mesh_%d" % loose["vb"]
        data = self.make_mesh(loose, name)
        if data is None:
            return
        obj = bpy.data.objects.new(name, data)
        self.coll.objects.link(obj)
        self.created.append(obj)
        self.warn("No Mesh node for vertex buffer %d: built '%s' from loose geometry"
                  % (loose["vb"], name))

    def make_empty(self, prefix, idx):
        o = bpy.data.objects.new("%s_%d" % (prefix, idx), None)
        o.empty_display_type = 'PLAIN_AXES'
        o.empty_display_size = 0.25
        return o

    def build(self, idx, parent, parent_inv=None):
        if idx in self.done:
            return
        d = self.m3g.get(idx)
        if d is None:
            return
        if d["type"] == T_EXTERNAL:
            self.done.add(idx)
            self.build_external(idx, parent, parent_inv)
            return
        if d["type"] not in NODE_TYPES:
            return
        self.done.add(idx)
        t = d["type"]
        cam_like = t in (T_CAMERA, T_LIGHT)
        obj = None

        if t in (T_GROUP, T_WORLD):
            obj = self.make_empty("World" if t == T_WORLD else "Group", idx)
        elif t in (T_MESH, T_MORPHING_MESH, T_SKINNED_MESH):
            data = self.make_mesh(d, "Mesh_%d" % idx)
            if data is None:
                return
            obj = bpy.data.objects.new("Mesh_%d" % idx, data)
        elif t == T_CAMERA:
            obj = bpy.data.objects.new("Camera_%d" % idx, self.make_camera(d, idx))
        elif t == T_LIGHT:
            ld = self.make_light(d, idx)
            if ld is None:
                obj = self.make_empty("Ambient", idx)
                cam_like = False
            else:
                obj = bpy.data.objects.new("Light_%d" % idx, ld)
        else:                      # sprite
            obj = self.make_empty("Sprite", idx)

        # Animated node with a general matrix: composite = T R S M. Animation
        # replaces T/R/S, so M goes into the children's parent-inverse instead.
        trs_anim = self.do_anim and self.has_trs_anim(d)
        split_m = bool(trs_anim and d["M"] is not None and not is_identity16(d["M"]))
        child_pinv = None
        if split_m:
            if t in (T_GROUP, T_WORLD):
                child_pinv = self.m_conj(d["M"])
            else:
                self.warn("Node %d: general transform ignored (node is animated)" % idx)

        self.coll.objects.link(obj)
        if parent is not None:
            obj.parent = parent
            if parent_inv is not None:
                obj.matrix_parent_inverse = parent_inv
        if trs_anim:
            obj.rotation_mode = 'QUATERNION'
        obj.matrix_basis = self.node_matrix(d, cam_like, skip_m=split_m)
        if d["uid"]:
            obj["m3g_user_id"] = str(d["uid"])     # str: may exceed int32
        for pid, val in d["params"]:
            obj["m3g_param_%d" % pid] = param_text(val)
        if obj.type == 'EMPTY':
            self.empties.append(obj)
        if not d["render"]:
            obj.hide_render = True
            obj.hide_viewport = True
        self.created.append(obj)
        self.node_objs[idx] = obj

        if t in (T_GROUP, T_WORLD):
            for c in d["children"]:
                self.build(c, obj, child_pinv)

    def build_external(self, idx, parent, parent_inv):
        if self.depth >= 8:
            self.warn("External references nested too deep, skipped")
            return
        ref = self.m3g.resolve_node(idx)
        if ref is None:
            return
        sub = Importer(ref[0], self.context, self.z_up, self.scale, self.name,
                       parent=self, import_animation=self.do_anim,
                       import_background=False)
        self.subs.append(sub)
        sub.build(ref[1], parent, parent_inv)

    # ---- camera / light -------------------------------------------------

    def make_camera(self, d, idx):
        cam = bpy.data.cameras.new("Camera_%d" % idx)
        cam.sensor_fit = 'VERTICAL'
        proj = d["proj"]
        if proj == 49:
            cam.type = 'ORTHO'
            cam.ortho_scale = max(d["fovy"], 1e-6) * self.scale
        else:
            cam.type = 'PERSP'
            if proj == 50:
                fov = min(max(d["fovy"], 0.1), 179.0)
                cam.angle_y = math.radians(fov)
            else:
                self.warn("Camera %d: generic projection matrix not supported, using default" % idx)
        if proj != 48:
            cam.clip_start = max(d["near"] * self.scale, 1e-4)
            cam.clip_end = max(d["far"] * self.scale, cam.clip_start + 1e-3)
        return cam

    def make_light(self, d, idx):
        mode = d["mode"]
        if mode == 129:
            lt = 'SUN'
        elif mode == 130:
            lt = 'POINT'
        elif mode == 131:
            lt = 'SPOT'
        else:
            return None
        L = bpy.data.lights.new("Light_%d" % idx, lt)
        L.color = rgb_lin(d["color"])
        inten = max(d["intensity"], 0.0)
        L.energy = inten if lt == 'SUN' else 1000.0 * inten
        if lt == 'SPOT':
            L.spot_size = math.radians(min(max(d["spot_angle"] * 2.0, 1.0), 180.0))
            L.spot_blend = 0.15
        return L

    # ---- mesh -----------------------------------------------------------

    def conv(self, a):
        if self.z_up:
            return np.stack((a[:, 0], -a[:, 2], a[:, 1]), axis=1)
        return a

    def make_mesh(self, d, name):
        m3g = self.m3g
        key = (d["vb"], tuple(d["subs"]))
        if key in self.mesh_cache:
            return self.mesh_cache[key]

        vb = m3g.get(d["vb"], T_VERTEX_BUFFER)
        if vb is None:
            self.warn("%s: vertex buffer missing/unreadable" % name)
            return None
        pva = m3g.get(vb["positions"], T_VERTEX_ARRAY)
        if pva is None:
            self.warn("%s: position array missing/unreadable" % name)
            return None

        raw = pva["data"].astype(np.float64)
        nv = raw.shape[0]
        if raw.shape[1] < 3:
            raw = np.concatenate([raw, np.zeros((nv, 3 - raw.shape[1]))], axis=1)
        P = raw[:, :3] * vb["pscale"] + np.asarray(vb["pbias"], dtype=np.float64)
        P = self.conv(P) * self.scale

        mesh = bpy.data.meshes.new(name)
        self.mesh_cache[key] = mesh

        tri_list, mat_list, smooth_list = [], [], []
        slot_of = {}
        for ib_idx, app_idx in d["subs"]:
            ib = m3g.get(ib_idx, T_STRIP_ARRAY)
            if ib is None:
                self.warn("%s: a submesh index buffer is missing/unreadable" % name)
                continue
            app = m3g.get(app_idx, T_APPEARANCE) if app_idx else None
            pm = m3g.get(app["polygon_mode"], T_POLYGON_MODE) if app and app["polygon_mode"] else None
            cw = bool(pm is not None and pm["winding"] == 169)
            smooth = not (pm is not None and pm["shading"] == 164)
            tris = strip_triangles(ib["indices"], ib["lengths"], cw, nv)
            if len(tris) == 0:
                continue
            slot = slot_of.get(app_idx)
            if slot is None:
                slot = len(slot_of)
                slot_of[app_idx] = slot
                mesh.materials.append(self.get_material(app_idx))
            tri_list.append(tris)
            mat_list.append(np.full(len(tris), slot, dtype=np.int64))
            smooth_list.append(np.full(len(tris), smooth, dtype=bool))

        faces = np.concatenate(tri_list).tolist() if tri_list else []
        mesh.from_pydata(P.tolist(), [], faces)
        mesh.update()

        if not faces or len(mesh.polygons) != len(faces):
            return mesh

        mat_idx = np.concatenate(mat_list)
        smooth = np.concatenate(smooth_list)
        set_poly_prop(mesh, "material_index", mat_idx.tolist())
        if smooth.any():
            set_poly_prop(mesh, "use_smooth", smooth.tolist())

        # normals
        nva = m3g.get(vb["normals"], T_VERTEX_ARRAY) if vb["normals"] else None
        if (nva is not None and smooth.any() and nva["data"].shape[0] == nv
                and nva["data"].shape[1] >= 3):
            N = self.conv(nva["data"][:, :3].astype(np.float64))
            ln = np.linalg.norm(N, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            N = N / ln
            try:
                mesh.normals_split_custom_set_from_vertices(N.tolist())
            except Exception as e:
                self.warn("%s: custom normals failed: %s" % (name, e))

        loop_vi = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vi)

        # UVs (M3G v origin is top, Blender is bottom -> flip v; images flipped too)
        for k, (tc_idx, bias, scale) in enumerate(vb["texcoords"]):
            va = m3g.get(tc_idx, T_VERTEX_ARRAY)
            if va is None or va["data"].shape[0] != nv or va["data"].shape[1] < 2:
                continue
            uv = va["data"][:, :2].astype(np.float64) * scale + np.asarray(bias[:2], dtype=np.float64)
            uv[:, 1] = 1.0 - uv[:, 1]
            try:
                uvl = mesh.uv_layers.new(name=uv_name(k))
            except Exception:
                break
            if uvl is None:
                break
            uvl.data.foreach_set("uv", uv[loop_vi].astype(np.float32).ravel())

        # vertex colors
        cva = m3g.get(vb["colors"], T_VERTEX_ARRAY) if vb["colors"] else None
        if cva is not None and cva["data"].shape[0] == nv and cva["data"].shape[1] in (3, 4):
            rawc = cva["data"]
            if cva["csize"] == 1:
                c = (rawc & 0xFF) / 255.0
            else:
                c = (rawc & 0xFFFF) / 65535.0
            rgba = np.ones((nv, 4), dtype=np.float64)
            rgba[:, :c.shape[1]] = c
            rgba[:, :3] = lin_np(rgba[:, :3])
            try:
                attr = mesh.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
                attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
            except Exception as e:
                self.warn("%s: vertex colors failed: %s" % (name, e))

        return mesh

    # ---- materials / images ---------------------------------------------

    def get_material(self, app_idx):
        mat = self.mat_cache.get(app_idx)
        if mat is not None:
            return mat
        m3g = self.m3g
        app = m3g.get(app_idx, T_APPEARANCE) if app_idx else None
        mat = bpy.data.materials.new("M3G_Material_%d" % app_idx if app_idx else "M3G_Default")
        self.mat_cache[app_idx] = mat
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is None:
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if app is None:
            return mat

        md = m3g.get(app["material"], T_MATERIAL) if app["material"] else None
        comp = m3g.get(app["compositing"], T_COMPOSITING) if app["compositing"] else None
        pm = m3g.get(app["polygon_mode"], T_POLYGON_MODE) if app["polygon_mode"] else None
        use_alpha = comp is not None and comp["blending"] in (64, 65)

        def setin(name, value):
            s = bsdf.inputs.get(name)
            if s is not None:
                s.default_value = value

        base = (0.8, 0.8, 0.8, 1.0)
        alpha = 1.0
        if md is not None:
            base = rgb_lin(md["diffuse"]) + (1.0,)
            if use_alpha:
                alpha = md["diffuse"][3] / 255.0
            sh = min(max(md["shininess"], 0.0), 128.0)
            sp = md["specular"]
            em = md["emissive"]
            setin("Roughness", math.sqrt(2.0 / (sh + 2.0)))
            setin("Specular IOR Level", min(1.0, (sp[0] + sp[1] + sp[2]) / (3 * 255.0)))
            if em[0] or em[1] or em[2]:
                setin("Emission Color", rgb_lin(em) + (1.0,))
                setin("Emission Strength", 1.0)
        setin("Base Color", base)
        setin("Alpha", alpha)
        mat.diffuse_color = (base[0], base[1], base[2], alpha)

        culling = pm["culling"] if pm is not None else 160   # M3G default: cull back
        mat.use_backface_culling = (culling == 160)

        if use_alpha:
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = 'BLENDED'
            elif hasattr(mat, "blend_method"):
                mat.blend_method = 'BLEND'

        self.hook_texture(nt, bsdf, app, base, use_alpha)
        return mat

    def hook_texture(self, nt, bsdf, app, base, use_alpha):
        m3g = self.m3g
        found = None
        for unit, ti in enumerate(app["textures"]):
            t = m3g.get(ti, T_TEXTURE2D)
            if t is not None and t["image"]:
                ref = m3g.resolve_image(t["image"])
                info = self.get_image(ref[1], ref[0]) if ref is not None else None
                if info is not None:
                    found = (unit, t, info)
                    break
        if found is None:
            return
        unit, t, info = found
        img = info["img"]
        has_alpha = info["alpha"]

        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uv_name(unit)
        uvn.location = (-900, 300)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.location = (-650, 300)
        tex.extension = 'EXTEND' if (t["wrap_s"] == 240 and t["wrap_t"] == 240) else 'REPEAT'
        tex.interpolation = 'Closest' if t["image_filter"] == 210 else 'Linear'
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

        white = base[0] > 0.999 and base[1] > 0.999 and base[2] > 0.999
        if t["blending"] == 227 and not white:     # MODULATE with diffuse
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.location = (-350, 300)
            mix.inputs[0].default_value = 1.0
            mix.inputs[7].default_value = base
            nt.links.new(tex.outputs["Color"], mix.inputs[6])
            nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if use_alpha and has_alpha:
            nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])

    def get_image(self, idx, file=None):
        file = file or self.m3g
        key = (id(file), idx)
        if key in self.img_cache:
            return self.img_cache[key]
        self.img_cache[key] = None
        d = file.get(idx, T_IMAGE2D)
        if d is None or d["mutable"]:
            return None
        fmt = d["format"]
        bpp = {96: 1, 97: 1, 98: 2, 99: 3, 100: 4}.get(fmt)
        w, h = d["width"], d["height"]
        if bpp is None or w == 0 or h == 0:
            self.warn("Image %d: unsupported format/size" % idx)
            return None
        n = w * h
        pix = np.frombuffer(d["pixels"], dtype=np.uint8)
        if d["palette"]:
            pal = np.frombuffer(d["palette"], dtype=np.uint8)
            if len(pal) == 0 or len(pal) % bpp != 0 or len(pix) < n:
                self.warn("Image %d: bad palette/pixel data" % idx)
                return None
            pal = pal.reshape(-1, bpp)
            ix = np.minimum(pix[:n].astype(np.int64), len(pal) - 1)
            px = pal[ix]
        else:
            if len(pix) < n * bpp:
                self.warn("Image %d: pixel data too short" % idx)
                return None
            px = pix[:n * bpp].reshape(n, bpp)

        rgba = np.empty((n, 4), dtype=np.uint8)
        if fmt == 96:
            rgba[:, :3] = 255
            rgba[:, 3] = px[:, 0]
        elif fmt == 97:
            rgba[:, :3] = px[:, :1]
            rgba[:, 3] = 255
        elif fmt == 98:
            rgba[:, :3] = px[:, :1]
            rgba[:, 3] = px[:, 1]
        elif fmt == 99:
            rgba[:, :3] = px
            rgba[:, 3] = 255
        else:
            rgba[:] = px
        # M3G rows are top-to-bottom, Blender bottom-to-top
        arr = (rgba.reshape(h, w, 4)[::-1].astype(np.float32) / 255.0).ravel()
        try:
            img = bpy.data.images.new("M3G_Image_%d" % idx, w, h, alpha=True)
            img.pixels.foreach_set(arr)
            img.update()
            try:
                img.pack()
            except Exception:
                pass
        except Exception as e:
            self.warn("Image %d: could not create (%s)" % (idx, e))
            return None
        res = {"img": img, "alpha": fmt in (96, 98, 100), "w": w, "h": h}
        self.img_cache[key] = res
        return res

    # ---- world background -------------------------------------------------

    def setup_background(self):
        """M3G Background -> Blender world: color + screen-space image."""
        m3g = self.m3g
        widx = next((i for i in sorted(m3g.objects) if m3g.objects[i][0] == T_WORLD), None)
        if widx is None:
            return
        wd = m3g.get(widx)
        if wd is None or wd["type"] != T_WORLD or not wd["background"]:
            return
        bgd = m3g.get(wd["background"], T_BACKGROUND)
        if bgd is None:
            return
        world = bpy.data.worlds.new("M3G_World")
        world.use_nodes = True
        nt = world.node_tree
        bgn = next((n for n in nt.nodes if n.type == 'BACKGROUND'), None)
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_WORLD'), None)
        if bgn is None:
            bgn = nt.nodes.new("ShaderNodeBackground")
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputWorld")
            nt.links.new(bgn.outputs["Background"], out.inputs["Surface"])
        color = rgb_lin(bgd["color"]) + (1.0,)
        bgn.inputs["Color"].default_value = color
        state = {"d": bgd, "nt": nt, "mapping": None, "size": None,
                 "crop": bgd["crop"], "color_sock": (bgn, bgn.inputs["Color"])}
        if bgd["image"]:
            ref = m3g.resolve_image(bgd["image"])
            info = self.get_image(ref[1], ref[0]) if ref is not None else None
            if info is not None:
                self.bg_image_nodes(nt, bgn, bgd, info, color, state)
        self.bg = state
        self.context.scene.world = world

    def range_mask(self, nt, sock):
        """1.0 where 0 < sock < 1, else 0.0 (Math nodes)."""
        nodes, links = nt.nodes, nt.links
        g = nodes.new("ShaderNodeMath")
        g.operation = 'GREATER_THAN'
        g.inputs[1].default_value = 0.0
        lo = nodes.new("ShaderNodeMath")
        lo.operation = 'LESS_THAN'
        lo.inputs[1].default_value = 1.0
        m = nodes.new("ShaderNodeMath")
        m.operation = 'MULTIPLY'
        links.new(sock, g.inputs[0])
        links.new(sock, lo.inputs[0])
        links.new(g.outputs[0], m.inputs[0])
        links.new(lo.outputs[0], m.inputs[1])
        return m

    def bg_image_nodes(self, nt, bgn, bgd, info, color, state):
        # Image drawn behind the scene; crop rect is in image pixels (origin
        # top-left) and is stretched over the whole frame. Window coords (0..1,
        # origin bottom-left) -> Mapping -> image UV. BORDER mode -> color outside.
        cx, cy, cw, ch = bgd["crop"]
        w, h = info["w"], info["h"]
        if cw == 0:
            cw = w
        if ch == 0:
            ch = h
        state["crop"] = (cx, cy, cw, ch)
        state["size"] = (w, h)
        nodes, links = nt.nodes, nt.links
        tc = nodes.new("ShaderNodeTexCoord")
        mp = nodes.new("ShaderNodeMapping")
        mp.vector_type = 'POINT'
        mp.inputs["Location"].default_value = (cx / w, (h - cy - ch) / h, 0.0)
        mp.inputs["Scale"].default_value = (cw / w, ch / h, 1.0)
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = info["img"]
        tex.interpolation = 'Closest'
        tex.extension = 'REPEAT'
        links.new(tc.outputs["Window"], mp.inputs["Vector"])
        links.new(mp.outputs["Vector"], tex.inputs["Vector"])

        flags = ((bgd["mode_x"] == 32, 'X'), (bgd["mode_y"] == 32, 'Y'))   # 32 = BORDER
        masks = []
        if any(f for f, _a in flags):
            sep = nodes.new("ShaderNodeSeparateXYZ")
            links.new(mp.outputs["Vector"], sep.inputs["Vector"])
            for f, axis in flags:
                if f:
                    masks.append(self.range_mask(nt, sep.outputs[axis]))
        mask = None
        if masks:
            mask = masks[0]
            if len(masks) == 2:
                mul = nodes.new("ShaderNodeMath")
                mul.operation = 'MULTIPLY'
                links.new(masks[0].outputs[0], mul.inputs[0])
                links.new(masks[1].outputs[0], mul.inputs[1])
                mask = mul
        if mask is None:
            links.new(tex.outputs["Color"], bgn.inputs["Color"])
        else:
            mix = nodes.new("ShaderNodeMix")
            mix.data_type = 'RGBA'
            mix.blend_type = 'MIX'
            mix.inputs[6].default_value = color
            links.new(mask.outputs[0], mix.inputs[0])
            links.new(tex.outputs["Color"], mix.inputs[7])
            links.new(mix.outputs[2], bgn.inputs["Color"])
            state["color_sock"] = (mix, mix.inputs[6])
        state["mapping"] = mp

    # ---- animation --------------------------------------------------------
    # One Blender action per AnimationController ("clip"). The first clip of
    # each target is assigned, the others are kept with a fake user so they
    # can be picked in the Action editor. Controller weight / active interval
    # are ignored; speed and reference times are applied.

    def group_tracks(self, track_ids):
        groups = {}
        for tid in track_ids:
            tr = self.m3g.get(tid, T_TRACK)
            if tr is None:
                continue
            seq = self.m3g.get(tr["seq"], T_KEYFRAMES)
            if seq is None or len(seq["times"]) == 0:
                continue
            ctrl = self.m3g.get(tr["ctrl"], T_CONTROLLER) if tr["ctrl"] else None
            groups.setdefault(tr["ctrl"], []).append((tr, seq, ctrl))
        return groups

    def keys(self, seq, ctrl):
        """-> frames, values (n,c), blender interpolation, loop flag."""
        n = len(seq["times"])
        a, b = seq["first"], seq["last"]
        if not (0 <= a <= b < n):
            a, b = 0, n - 1
        times = seq["times"][a:b + 1].astype(np.float64)
        vals = seq["values"][a:b + 1].copy()
        loop = seq["repeat"] == 193
        dur = float(seq["duration"])
        if loop and dur > 0.0 and times[0] + dur > times[-1] + 1e-9:
            # M3G loops by interpolating last key -> first key shifted by duration
            times = np.append(times, times[0] + dur)
            vals = np.vstack([vals, vals[0:1]])
        speed = ctrl["speed"] if (ctrl is not None and ctrl["speed"] > 0.0) else 1.0
        refs = ctrl["refseq"] if ctrl is not None else 0.0
        refw = ctrl["refworld"] if ctrl is not None else 0
        world_ms = refw + (times - refs) / speed
        frames = self.frame0 + world_ms * (self.fps / 1000.0)
        self.anim_state["last"] = max(self.anim_state["last"], float(frames[-1]))
        return frames, vals, INTERP.get(seq["interp"], 'LINEAR'), loop

    def add_fcurve(self, act, path, index, frames, vals, interp, loop):
        try:
            fc = act.fcurves.new(data_path=path, index=index)
        except RuntimeError:
            self.warn("Duplicate animation channel %s[%d] skipped" % (path, index))
            return
        n = len(frames)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, dtype=np.float32)
        co[0::2] = frames
        co[1::2] = vals
        fc.keyframe_points.foreach_set("co", co)
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            if interp == 'BEZIER':
                kp.handle_left_type = 'AUTO'
                kp.handle_right_type = 'AUTO'
        fc.update()
        if loop:
            fc.modifiers.new('CYCLES')

    def run_clips(self, name, ids, groups, channel_fn):
        assigned = set()
        for cidx in sorted(groups):
            acts = {}
            for tr, seq, ctrl in groups[cidx]:
                for (tk, path, index, frames, vals, interp, loop) in channel_fn(tr, seq, ctrl):
                    if tk not in ids:
                        continue
                    act = acts.get(tk)
                    if act is None:
                        act = bpy.data.actions.new(
                            "M3G_%s_c%d%s" % (name, cidx, "" if tk == 'obj' else "_" + tk))
                        acts[tk] = act
                    self.add_fcurve(act, path, index, frames, vals, interp, loop)
            for tk, act in acts.items():
                if tk not in assigned:
                    assign_action(ids[tk], act)
                    assigned.add(tk)
                else:
                    act.use_fake_user = True

    def node_channels(self, d, obj, tr, seq, ctrl):
        prop = tr["prop"]
        t = d["type"]
        cam_like = t in (T_CAMERA, T_LIGHT)
        frames, vals, interp, loop = self.keys(seq, ctrl)
        c = vals.shape[1]
        out = []

        def add(target, path, arrays):
            for i, a in enumerate(arrays):
                out.append((target, path, i, frames, a, interp, loop))

        data = obj.data
        if prop == P_TRANSLATION and c >= 3:
            loc = self.conv(vals[:, :3]) * self.scale
            add('obj', "location", [loc[:, 0], loc[:, 1], loc[:, 2]])
        elif prop == P_ORIENTATION and c >= 4:
            q = vals[:, [3, 0, 1, 2]].copy()           # (x,y,z,w) -> (w,x,y,z)
            nrm = np.linalg.norm(q, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            q = q / nrm
            if self.z_up:
                h = math.sqrt(0.5)
                qc = np.tile([h, h, 0.0, 0.0], (len(q), 1))
                qci = np.tile([h, -h, 0.0, 0.0], (len(q), 1))
                q = qmul(qc, q)
                if not cam_like:
                    q = qmul(q, qci)
            for i in range(1, len(q)):
                if np.dot(q[i - 1], q[i]) < 0.0:
                    q[i] = -q[i]
            add('obj', "rotation_quaternion", [q[:, k] for k in range(4)])
        elif prop == P_SCALE and c >= 1:
            sc = np.repeat(vals[:, :1], 3, axis=1) if c < 3 else vals[:, :3]
            if self.z_up and not cam_like:
                sc = sc[:, [0, 2, 1]]
            add('obj', "scale", [sc[:, 0], sc[:, 1], sc[:, 2]])
        elif t == T_CAMERA and prop == P_FOV and c >= 1 and data is not None:
            sh = getattr(data, "sensor_height", 24.0)
            fov = np.clip(vals[:, 0], 0.1, 179.0)
            add('data', "lens", [sh / (2.0 * np.tan(np.radians(fov) / 2.0))])
        elif t == T_CAMERA and prop == P_NEAR and c >= 1:
            add('data', "clip_start", [np.maximum(vals[:, 0] * self.scale, 1e-4)])
        elif t == T_CAMERA and prop == P_FAR and c >= 1:
            add('data', "clip_end", [vals[:, 0] * self.scale])
        elif t == T_LIGHT and prop == P_COLOR and c >= 3 and data is not None:
            col = lin_np(np.clip(vals[:, :3], 0.0, 1.0))
            add('data', "color", [col[:, 0], col[:, 1], col[:, 2]])
        elif t == T_LIGHT and prop == P_INTENSITY and c >= 1 and data is not None:
            k = 1.0 if getattr(data, "type", "") == 'SUN' else 1000.0
            add('data', "energy", [np.maximum(vals[:, 0], 0.0) * k])
        elif t == T_LIGHT and prop == P_SPOT_ANGLE and c >= 1 and data is not None:
            add('data', "spot_size", [np.radians(np.clip(vals[:, 0] * 2.0, 1.0, 180.0))])
        else:
            self.warn_once(("prop", prop, t),
                           "Animation property %d on node type %d not supported" % (prop, t))
        return out

    def bg_channels(self, tr, seq, ctrl):
        bg = self.bg
        prop = tr["prop"]
        frames, vals, interp, loop = self.keys(seq, ctrl)
        c = vals.shape[1]
        out = []
        base = 'nodes["%s"].inputs[%d].default_value'
        if prop == P_CROP and bg["mapping"] is not None and c >= 2:
            w, h = bg["size"]
            cx, cy = vals[:, 0], vals[:, 1]
            cw = vals[:, 2] if c >= 4 else np.full(len(cx), float(bg["crop"][2]))
            ch = vals[:, 3] if c >= 4 else np.full(len(cx), float(bg["crop"][3]))
            mp = bg["mapping"]
            li = socket_index(mp, mp.inputs["Location"])
            si = socket_index(mp, mp.inputs["Scale"])
            out.append(('nt', base % (mp.name, li), 0, frames, cx / w, interp, loop))
            out.append(('nt', base % (mp.name, li), 1, frames, (h - cy - ch) / h, interp, loop))
            if c >= 4:
                out.append(('nt', base % (mp.name, si), 0, frames, cw / w, interp, loop))
                out.append(('nt', base % (mp.name, si), 1, frames, ch / h, interp, loop))
        elif prop == P_COLOR and c >= 3:
            node, sock = bg["color_sock"]
            si = socket_index(node, sock)
            col = lin_np(np.clip(vals[:, :3], 0.0, 1.0))
            for i in range(3):
                out.append(('nt', base % (node.name, si), i, frames, col[:, i], interp, loop))
            if c >= 4:
                out.append(('nt', base % (node.name, si), 3, frames, vals[:, 3], interp, loop))
        elif prop == P_CROP:
            self.warn_once(("bgcrop",), "Background crop animation skipped (no background image loaded)")
        else:
            self.warn_once(("bgprop", prop), "Background animation property %d not supported" % prop)
        return out

    def animate_node(self, idx, obj):
        d = self.m3g.get(idx)
        if d is None or not d.get("tracks"):
            return
        groups = self.group_tracks(d["tracks"])
        if not groups:
            return
        ids = {'obj': obj}
        if d["type"] in (T_CAMERA, T_LIGHT) and obj.data is not None:
            ids['data'] = obj.data
        self.run_clips(obj.name, ids, groups,
                       lambda tr, seq, ctrl: self.node_channels(d, obj, tr, seq, ctrl))

    def animate_all(self):
        for idx, obj in list(self.node_objs.items()):
            try:
                self.animate_node(idx, obj)
            except Exception as e:
                self.warn("Animation of node %d failed: %s" % (idx, e))
        if self.bg is not None and self.bg["d"].get("tracks"):
            try:
                groups = self.group_tracks(self.bg["d"]["tracks"])
                self.run_clips("Background", {'nt': self.bg["nt"]}, groups, self.bg_channels)
            except Exception as e:
                self.warn("Background animation failed: %s" % e)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class IMPORT_OT_m3g(bpy.types.Operator, ImportHelper):
    """Import a Mobile 3D Graphics (JSR-184) file"""
    bl_idname = "import_scene.m3g"
    bl_label = "Import M3G"
    bl_options = {'UNDO'}

    filename_ext = ".m3g"
    filter_glob: StringProperty(default="*.m3g", options={'HIDDEN'})
    global_scale: FloatProperty(
        name="Scale", description="Uniform scale applied to the whole scene",
        default=1.0, min=1e-6, max=1e6)
    y_up_to_z_up: BoolProperty(
        name="Y-up to Z-up", description="Convert M3G (Y-up) to Blender (Z-up)",
        default=True)
    import_animation: BoolProperty(
        name="Animation",
        description="Import keyframe animation (one action per controller)",
        default=True)
    import_background: BoolProperty(
        name="Background",
        description="Set the scene World from the M3G Background (color + image)",
        default=True)
    load_external: BoolProperty(
        name="Load external files",
        description="Load files referenced by the .m3g if they sit next to it",
        default=True)

    def execute(self, context):
        try:
            if context.object is not None and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        try:
            with open(self.filepath, "rb") as f:
                data = f.read()
            m3g = M3GFile(data, path=self.filepath, allow_external=self.load_external)
            imp = Importer(m3g, context, self.y_up_to_z_up, self.global_scale,
                           os.path.splitext(os.path.basename(self.filepath))[0],
                           import_animation=self.import_animation,
                           import_background=self.import_background)
            count = imp.run()
        except M3GError as e:
            self.report({'ERROR'}, "M3G: %s" % e)
            return {'CANCELLED'}
        except OSError as e:
            self.report({'ERROR'}, "Cannot read file: %s" % e)
            return {'CANCELLED'}
        for note in m3g.notes:
            self.report({'INFO'}, note)
        for w in m3g.warnings[:10]:
            self.report({'WARNING'}, w)
        if len(m3g.warnings) > 10:
            self.report({'WARNING'}, "... and %d more warnings" % (len(m3g.warnings) - 10))
        self.report({'INFO'}, "Imported %d objects" % count)
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_m3g.bl_idname, text="M3G (.m3g)")


classes = (IMPORT_OT_m3g,)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
