"""Build a point-cloud conditioning scene from scene.json, and render it.

    blender -b --factory-startup --python cond_build.py -- \
        --scene <scene.json> --save <out.blend> --render <dir> \
        --width 832 --height 480 --frames 61 --point_px 4

The render is points only, on black, at the target resolution. Frames 1 and N
are deliberately NOT rendered -- the driver script copies the original footage
in there instead -- so this renders 2..N-1 only.

Two things make this different from an ordinary point-cloud render:

  * Every point is the same size on screen. A fixed world radius would shrink
    with distance; instead each quad is scaled by its own depth from the camera,
    size = point_px * depth / focal_px, so it lands on exactly point_px pixels.
  * The quads are square and screen-aligned: they are instanced flat and given
    the rotation of the camera, so they stay parallel to the image plane and
    project to squares rather than foreshortened diamonds.
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector

DEFAULTS = {
    "scene": None, "images": None, "save": None, "render": None,
    "width": "832", "height": "480", "frames": "61", "point_px": "4",
    "span": "0", "footage": "1",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = dict(DEFAULTS)
    key = None
    for tok in argv:
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
        elif key:
            args[key] = tok
            key = None
    if not args["scene"]:
        raise SystemExit("--scene <scene.json> is required")
    for k in ("width", "height", "frames"):
        args[k] = int(args[k])
    args["point_px"] = float(args["point_px"])
    args["span"] = args["span"] not in ("0", "false", "no", "")
    args["footage"] = args["footage"] not in ("0", "false", "no", "")
    return args


def use_nodes(mat):
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat.node_tree


def pick_frames(all_frames, n, span):
    if len(all_frames) <= n:
        return list(all_frames)
    if span:
        idx = [round(i * (len(all_frames) - 1) / (n - 1)) for i in range(n)]
        return [all_frames[i] for i in idx]
    return list(all_frames[:n])


def build_points(data, coll):
    xyz = data["points"]["xyz"]
    rgb = data["points"]["rgb"]

    mesh = bpy.data.meshes.new("SparsePoints")
    mesh.from_pydata([Vector(p) for p in xyz], [], [])
    mesh.update()

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

    mat = bpy.data.materials.new("PointSquareMat")
    nt = use_nodes(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "Col"
    nt.links.new(col.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj, mat


def build_squares_modifier(obj, mat, cam, size_per_unit_depth):
    """Screen-aligned squares whose on-screen size does not change with depth."""
    ng = bpy.data.node_groups.new("PointSquares", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT",
                            socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT",
                            socket_type="NodeSocketGeometry")
    n = ng.nodes.new
    gin, gout = n("NodeGroupInput"), n("NodeGroupOutput")

    # The camera, read in the space of the point object: geometry nodes work in
    # local coordinates, and both objects hang off the same root.
    info = n("GeometryNodeObjectInfo")
    info.transform_space = "RELATIVE"
    info.inputs["Object"].default_value = cam

    pos = n("GeometryNodeInputPosition")
    delta = n("ShaderNodeVectorMath")
    delta.operation = "SUBTRACT"
    fwd = n("ShaderNodeVectorRotate")
    fwd.rotation_type = "EULER_XYZ"
    fwd.inputs["Vector"].default_value = (0.0, 0.0, -1.0)   # cameras look down -Z
    depth = n("ShaderNodeVectorMath")
    depth.operation = "DOT_PRODUCT"
    size = n("ShaderNodeMath")
    size.operation = "MULTIPLY"
    size.inputs[1].default_value = size_per_unit_depth

    quad = n("GeometryNodeMeshGrid")
    quad.inputs["Size X"].default_value = 1.0
    quad.inputs["Size Y"].default_value = 1.0
    quad.inputs["Vertices X"].default_value = 2
    quad.inputs["Vertices Y"].default_value = 2

    iop = n("GeometryNodeInstanceOnPoints")
    real = n("GeometryNodeRealizeInstances")
    setmat = n("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat

    for i, node in enumerate((gin, info, pos, delta, fwd, depth, size, quad,
                              iop, real, setmat, gout)):
        node.location = (-900 + 160 * i, 0)

    link = ng.links.new
    link(pos.outputs["Position"], delta.inputs[0])
    link(info.outputs["Location"], delta.inputs[1])
    link(info.outputs["Rotation"], fwd.inputs["Rotation"])
    link(delta.outputs["Vector"], depth.inputs[0])
    link(fwd.outputs["Vector"], depth.inputs[1])
    link(depth.outputs["Value"], size.inputs[0])

    link(gin.outputs[0], iop.inputs["Points"])
    link(quad.outputs["Mesh"], iop.inputs["Instance"])
    link(info.outputs["Rotation"], iop.inputs["Rotation"])
    link(size.outputs["Value"], iop.inputs["Scale"])
    link(iop.outputs["Instances"], real.inputs["Geometry"])
    link(real.outputs["Geometry"], setmat.inputs["Geometry"])
    link(setmat.outputs["Geometry"], gout.inputs[0])

    mod = obj.modifiers.new("PointSquares", "NODES")
    mod.node_group = ng
    return ng


def action_fcurves(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            bag = strip.channelbag(ad.action_slot)
            if bag:
                out.extend(bag.fcurves)
    return out


def build_camera(data, frames, coll):
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

    for i, f in enumerate(frames):
        m = Matrix(f["rotation_matrix"]).to_4x4()
        m.translation = Vector(f["location"])
        cam.matrix_world = m
        cam.keyframe_insert("location", frame=i + 1)
        cam.keyframe_insert("rotation_quaternion", frame=i + 1)

    for fc in action_fcurves(cam):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return cam


def build_footage_ref(data, frames, cam, coll, images_dir):
    """The original frames on a backdrop plane: viewport reference only."""
    p = data["plane"]
    hw, hh = p["width"] / 2.0, p["height"] / 2.0
    ox, oy, d = p["offset_x"], p["offset_y"], p["distance"]

    mesh = bpy.data.meshes.new("Footage_Ref")
    mesh.from_pydata([(ox - hw, oy - hh, -d), (ox + hw, oy - hh, -d),
                      (ox + hw, oy + hh, -d), (ox - hw, oy + hh, -d)],
                     [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = co

    obj = bpy.data.objects.new("Footage_Ref", mesh)
    coll.objects.link(obj)
    obj.parent = cam
    obj.matrix_parent_inverse = Matrix.Identity(4)

    img = bpy.data.images.load(os.path.join(images_dir, frames[0]["name"]),
                               check_existing=True)
    img.source = "SEQUENCE"

    mat = bpy.data.materials.new("Footage_RefMat")
    nt = use_nodes(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    tex.image_user.frame_duration = len(frames)
    tex.image_user.frame_start = 1
    tex.image_user.use_auto_refresh = True
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)

    obj.hide_render = True      # the render is points on black, nothing else
    return obj


def pick_engine():
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if cand in items:
            return cand
    return "CYCLES"


def main():
    args = parse_args()
    with open(args["scene"], "r") as fh:
        data = json.load(fh)
    images_dir = args["images"] or data["image_dir"]

    frames = pick_frames(data["frames"], args["frames"], args["span"])
    n = len(frames)
    if n < args["frames"]:
        print("warning: only {} solved frames available".format(n))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    coll = bpy.data.collections.new("COND")
    bpy.context.scene.collection.children.link(coll)

    c = data["camera"]
    rw, rh = args["width"], args["height"]
    W, H = c["width"], c["height"]
    fx, fy = c["fx"], c["fy"]

    points, mat = build_points(data, coll)
    cam = build_camera(data, frames, coll)
    if args["footage"]:
        build_footage_ref(data, frames, cam, coll, images_dir)

    # Horizontal focal length in render pixels, so a quad of this world size at
    # one unit of depth covers point_px pixels.
    focal_px = fx * rw / float(W)
    build_squares_modifier(points, mat, cam, args["point_px"] / focal_px)

    root = bpy.data.objects.new("COLMAP_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = data["plane"]["distance"] * 0.05
    coll.objects.link(root)
    if data.get("root_matrix"):
        root.matrix_world = Matrix(data["root_matrix"]).to_4x4()
    for obj in (points, cam):
        obj.parent = root
        obj.matrix_parent_inverse = Matrix.Identity(4)

    scene = bpy.context.scene
    scene.render.resolution_x = rw
    scene.render.resolution_y = rh
    scene.render.resolution_percentage = 100
    # Keep the solved vertical field of view even though 832x480 is not 16:9.
    # The footage frames get squeezed to the same shape, so the two still match.
    # Blender clamps both pixel aspects to >= 1, so a ratio below one has to be
    # expressed by widening X rather than narrowing Y.
    pa = (rw / float(rh)) * (H / float(W)) * (fx / fy)
    if pa >= 1.0:
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0, pa
    else:
        scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0 / pa, 1.0
    scene.render.fps = int(round(data.get("fps", 10)))
    scene.frame_start, scene.frame_end = 1, n

    world = bpy.data.worlds.new("Black")
    scene.world = world
    nt = use_nodes(world)
    for node in nt.nodes:
        if node.type == "BACKGROUND":
            node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    scene.render.engine = pick_engine()
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 16
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 16
    scene.render.filter_size = 0.6        # keep the little squares crisp
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"   # not AgX: keep photo colour
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    print("cond: {} frames, {} points, {}x{}, pixel aspect {:.5f}/{:.5f} "
          "(want {:.5f}), engine {}, {:.1f}px squares".format(
              n, data["points"]["count"], rw, rh,
              scene.render.pixel_aspect_x, scene.render.pixel_aspect_y, pa,
              scene.render.engine, args["point_px"]))

    if args["render"]:
        os.makedirs(args["render"], exist_ok=True)
        # The driver needs to know which source frames ended up as 1 and N, and
        # --span means that is not simply the first and the 61st.
        with open(os.path.join(args["render"], "manifest.json"), "w") as fh:
            json.dump({"image_dir": images_dir,
                       "width": rw, "height": rh,
                       "names": [f["name"] for f in frames]}, fh, indent=1)
        scene.render.filepath = os.path.join(args["render"], "frame_")
        # 1 and N are replaced by the real footage, so do not render them.
        scene.frame_start, scene.frame_end = 2, n - 1
        bpy.ops.render.render(animation=True)
        scene.frame_start, scene.frame_end = 1, n
        print("rendered frames 2..{} to {}".format(n - 1, args["render"]))

    scene.frame_set(1)
    if args["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=args["save"])
        print("saved " + args["save"])


main()
