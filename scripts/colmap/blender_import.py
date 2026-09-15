"""Build a Blender scene from scene.json (written by export_scene.py).

Run headless:
    blender.exe -b --factory-startup --python blender_import.py -- \
        --scene <scene.json> --images <dir> --save <out.blend>

or paste into Blender's Scripting tab and press Run (it will prompt-free import
using the paths baked into scene.json).

Creates, in a "COLMAP" collection:
  * Camera            -- animated, one keyframe per reconstructed frame
  * Footage           -- image plane parented to the camera, showing the frame
                         that camera actually saw
  * SparsePoints      -- the cloud, with photo colours, visible in renders
  * CameraPath        -- a polyline through the camera centres, for reference
  * COLMAP_Root       -- empty parent, so the whole solve can be moved at once
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {"scene": None, "images": None, "save": None, "point_radius": 0.0}
    key = None
    for tok in argv:
        if tok.startswith("--"):
            key = tok[2:]
        elif key:
            args[key] = tok
            key = None
    if not args["scene"]:
        raise SystemExit("--scene <scene.json> is required")
    args["point_radius"] = float(args["point_radius"] or 0.0)
    return args


def action_fcurves(obj):
    """F-curves of obj's action, across the legacy and slotted (4.4+) layouts."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    action = ad.action
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(ad.action_slot)
            if bag:
                out.extend(bag.fcurves)
    return out


def use_nodes(mat):
    """Enable the node tree without tripping the 5.x deprecation warning."""
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat.node_tree


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def build_points(data, coll, radius):
    xyz = data["points"]["xyz"]
    rgb = data["points"]["rgb"]

    mesh = bpy.data.meshes.new("SparsePoints")
    mesh.from_pydata([Vector(p) for p in xyz], [], [])
    mesh.update()

    # sRGB bytes -> linear scene referred, so the dots match the footage.
    def to_linear(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    flat = []
    for r, g, b in rgb:
        flat.extend((to_linear(r), to_linear(g), to_linear(b), 1.0))
    attr.data.foreach_set("color", flat)

    obj = bpy.data.objects.new("SparsePoints", mesh)
    coll.objects.link(obj)

    mat = bpy.data.materials.new("SparsePointsMat")
    nt = use_nodes(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "Col"
    nt.links.new(col.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)

    if radius <= 0.0:
        radius = float(data["points"].get("radius") or 0.0)
    if radius <= 0.0:
        lo = Vector((min(p[i] for p in xyz) for i in range(3)))
        hi = Vector((max(p[i] for p in xyz) for i in range(3)))
        radius = max((hi - lo).length * 0.0015, 1e-4)

    # Vertices alone render as nothing, so give them geometry via nodes.
    try:
        ng = bpy.data.node_groups.new("PointsToSpheres", "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT",
                                socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT",
                                socket_type="NodeSocketGeometry")
        gin = ng.nodes.new("NodeGroupInput")
        gout = ng.nodes.new("NodeGroupOutput")
        m2p = ng.nodes.new("GeometryNodeMeshToPoints")
        m2p.inputs["Radius"].default_value = radius
        setmat = ng.nodes.new("GeometryNodeSetMaterial")
        setmat.inputs["Material"].default_value = mat
        gin.location, m2p.location = (-400, 0), (-200, 0)
        setmat.location, gout.location = (0, 0), (200, 0)
        ng.links.new(gin.outputs[0], m2p.inputs["Mesh"])
        ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
        ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])
        mod = obj.modifiers.new("PointsToSpheres", "NODES")
        mod.node_group = ng
        print("points: radius {:.4f}".format(radius))
    except Exception as exc:  # keep the cloud even if the node API shifted
        print("geometry nodes setup failed ({}); mesh vertices only".format(exc))

    return obj


def build_camera(data, coll):
    c = data["camera"]
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = c["sensor_width"]
    cam_data.lens = c["lens_mm"]
    cam_data.shift_x = c["shift_x"]
    cam_data.shift_y = c["shift_y"]
    cam_data.clip_start = c.get("clip_start", 0.01)
    cam_data.clip_end = c.get("clip_end", 1000.0)
    cam_data.display_size = 0.5

    cam = bpy.data.objects.new("Camera", cam_data)
    cam.rotation_mode = "QUATERNION"
    coll.objects.link(cam)
    bpy.context.scene.camera = cam

    for f in data["frames"]:
        m = Matrix(f["rotation_matrix"]).to_4x4()
        m.translation = Vector(f["location"])
        cam.matrix_world = m
        cam.keyframe_insert("location", frame=f["scene_frame"])
        cam.keyframe_insert("rotation_quaternion", frame=f["scene_frame"])

    # Solved poses are samples, not a smooth intent: linear keeps the camera on
    # the measured path instead of letting bezier handles overshoot between keys.
    for fc in action_fcurves(cam):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return cam


def build_plane(data, cam, coll, images_dir):
    p = data["plane"]
    frames = data["frames"]
    hw, hh = p["width"] / 2.0, p["height"] / 2.0
    ox, oy, d = p["offset_x"], p["offset_y"], p["distance"]

    mesh = bpy.data.meshes.new("Footage")
    verts = [(ox - hw, oy - hh, -d), (ox + hw, oy - hh, -d),
             (ox + hw, oy + hh, -d), (ox - hw, oy + hh, -d)]
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    uv = mesh.uv_layers.new(name="UVMap")
    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = co

    obj = bpy.data.objects.new("Footage", mesh)
    coll.objects.link(obj)
    obj.parent = cam          # rides the camera, so it always fills the frame
    obj.matrix_parent_inverse = Matrix.Identity(4)

    first = os.path.join(images_dir, frames[0]["name"])
    img = bpy.data.images.load(first, check_existing=True)
    img.source = "SEQUENCE"

    mat = bpy.data.materials.new("FootageMat")
    nt = use_nodes(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    tex.image_user.frame_duration = len(frames)
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = 0
    tex.image_user.use_auto_refresh = True
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)

    obj.visible_shadow = False
    return obj


def build_path(data, coll):
    curve = bpy.data.curves.new("CameraPath", "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    pts = [f["location"] for f in data["frames"]]
    spline.points.add(len(pts) - 1)
    for sp, co in zip(spline.points, pts):
        sp.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new("CameraPath", curve)
    coll.objects.link(obj)
    return obj


def main():
    args = parse_args()
    with open(args["scene"], "r") as fh:
        data = json.load(fh)
    images_dir = args["images"] or data["image_dir"]

    clear_scene()
    coll = make_collection("COLMAP")

    points = build_points(data, coll, args["point_radius"])
    cam = build_camera(data, coll)
    plane = build_plane(data, cam, coll, images_dir)
    path = build_path(data, coll)

    root = bpy.data.objects.new("COLMAP_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = data["plane"]["distance"] * 0.05
    coll.objects.link(root)
    rm = data.get("root_matrix")
    if rm:
        root.matrix_world = Matrix(rm).to_4x4()
    for obj in (points, cam, path):
        obj.parent = root
        obj.matrix_parent_inverse = Matrix.Identity(4)

    scene = bpy.context.scene
    c = data["camera"]
    scene.render.resolution_x = c["width"]
    scene.render.resolution_y = c["height"]
    # Blender clamps both pixel aspects to >= 1, so a ratio below one has to be
    # expressed by widening X rather than narrowing Y.
    pa = c["pixel_aspect_y"]
    if pa >= 1.0:
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0, pa
    else:
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0 / pa, 1.0
    scene.render.fps = int(round(data["fps"]))
    scene.frame_start = data["frames"][0]["scene_frame"]
    scene.frame_end = data["frames"][-1]["scene_frame"]
    scene.frame_set(scene.frame_start)

    scene.world = bpy.data.worlds.new("World")

    print("built: {} frames, {} points, plane {:.3f} x {:.3f} at {:.3f}".format(
        len(data["frames"]), data["points"]["count"],
        data["plane"]["width"], data["plane"]["height"],
        data["plane"]["distance"]))

    if args["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=args["save"])
        print("saved " + args["save"])


main()
