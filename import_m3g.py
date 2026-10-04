# SPDX-License-Identifier: GPL-3.0-or-later
#
# M3G (JSR-184 / Mobile 3D Graphics) importer for Blender 4.5
#
# Imports: scene graph (World/Group), meshes (triangle strips, all vertex array
# encodings), normals, UVs, vertex colors, materials, textures (Image2D),
# cameras, lights.
# Not imported: animation, skinning/morph targets (mesh imported as static base
# shape), fog, background, sprites (empty), ambient light (empty).

bl_info = {
    "name": "M3G (JSR-184) Importer",
    "author": "Claude",
    "version": (1, 0, 0),
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

def _object3d(r):
    uid = r.u32()                 # userID
    for _ in range(r.u32()):      # animation tracks (not imported)
        r.u32()
    params = []
    for _ in range(r.u32()):      # user parameters
        pid = r.u32()
        params.append((pid, r.raw(r.u32())))
    return uid, params


def param_text(val):
    try:
        return val.decode("utf-8").replace("\x00", "")
    except UnicodeDecodeError:
        return "hex:" + val.hex()


def _transformable(r, d):
    d["uid"], d["params"] = _object3d(r)
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
    _object3d(r)
    r.u8()                        # layer
    d["compositing"] = r.u32()
    d["fog"] = r.u32()
    d["polygon_mode"] = r.u32()
    d["material"] = r.u32()
    n = r.u32()
    d["textures"] = [r.u32() for _ in range(n)]


def _h_material(r, d):
    _object3d(r)
    r.rgb()                       # ambient
    d["diffuse"] = r.rgba()
    d["emissive"] = r.rgb()
    d["specular"] = r.rgb()
    d["shininess"] = r.f32()
    d["track"] = r.boolean()


def _h_polygon_mode(r, d):
    _object3d(r)
    d["culling"] = r.u8()
    d["shading"] = r.u8()
    d["winding"] = r.u8()
    r.boolean()
    r.boolean()
    r.boolean()


def _h_compositing(r, d):
    _object3d(r)
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
    _object3d(r)
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
    _object3d(r)
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
    _object3d(r)
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
    _object3d(r)
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
}


# ---------------------------------------------------------------------------
# File container
# ---------------------------------------------------------------------------

class M3GFile:
    def __init__(self, data):
        self.objects = {}         # index -> (type, bytes)
        self.parsed = {}
        self.warnings = []
        self._load(data)

    def _load(self, data):
        if data[:12] != M3G_MAGIC:
            raise M3GError("Not an M3G file (bad JSR184 signature)")
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
    def __init__(self, m3g, context, z_up, scale, name):
        self.m3g = m3g
        self.context = context
        self.z_up = z_up
        self.scale = scale
        self.name = name
        # M3G is Y-up, camera looks down -Z. Blender is Z-up. C: (x,y,z)->(x,-z,y)
        self.C = Matrix.Rotation(math.radians(90.0), 4, 'X') if z_up else Matrix.Identity(4)
        self.Ci = self.C.inverted()
        self.coll = None
        self.done = set()
        self.node_objs = {}
        self.created = []
        self.empties = []
        self.mesh_cache = {}
        self.mat_cache = {}
        self.img_cache = {}

    def warn(self, msg):
        self.m3g.warnings.append(msg)

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

        for idx in sorted(m3g.objects):
            if m3g.objects[idx][0] == 255:
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

    def node_matrix(self, d, camera_like=False):
        T = Matrix.Translation(d["T"])
        R = Matrix.Identity(4)
        if d["R"] is not None:
            ang, ax = d["R"]
            v = Vector(ax)
            if v.length > 1e-12 and ang != 0.0:
                R = Matrix.Rotation(math.radians(ang), 4, v.normalized())
        S = Matrix.Diagonal((d["S"][0], d["S"][1], d["S"][2], 1.0))
        L = T @ R @ S
        if d["M"] is not None:
            m = d["M"]
            L = L @ Matrix((m[0:4], m[4:8], m[8:12], m[12:16]))
        if self.z_up:
            # cameras/lights point down local -Z in both systems -> C @ L
            L = (self.C @ L) if camera_like else (self.C @ L @ self.Ci)
        L.translation = L.translation * self.scale
        return L

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

    def build(self, idx, parent):
        if idx in self.done:
            return
        d = self.m3g.get(idx)
        if d is None or d["type"] not in NODE_TYPES:
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

        self.coll.objects.link(obj)
        if parent is not None:
            obj.parent = parent
        obj.matrix_basis = self.node_matrix(d, cam_like)
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
                self.build(c, obj)

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
                info = self.get_image(t["image"])
                if info is not None:
                    found = (unit, t, info)
                    break
        if found is None:
            return
        unit, t, (img, has_alpha) = found

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

    def get_image(self, idx):
        if idx in self.img_cache:
            return self.img_cache[idx]
        self.img_cache[idx] = None
        d = self.m3g.get(idx, T_IMAGE2D)
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
        res = (img, fmt in (96, 98, 100))
        self.img_cache[idx] = res
        return res


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

    def execute(self, context):
        try:
            if context.object is not None and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        try:
            with open(self.filepath, "rb") as f:
                data = f.read()
            m3g = M3GFile(data)
            imp = Importer(m3g, context, self.y_up_to_z_up, self.global_scale,
                           os.path.splitext(os.path.basename(self.filepath))[0])
            count = imp.run()
        except M3GError as e:
            self.report({'ERROR'}, "M3G: %s" % e)
            return {'CANCELLED'}
        except OSError as e:
            self.report({'ERROR'}, "Cannot read file: %s" % e)
            return {'CANCELLED'}
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
