import argparse
import gzip
import h5py
import json
import laspy
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs_py import point_cloud2

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from foxglove_msgs.msg import Grid, PackedElementField
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation, TextAnnotation, Point2, Color
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix, NavSatStatus
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray, Marker


def location_fru_to_flu(fru: list[3]):
    x = fru[0]
    y = fru[1]
    z = fru[2]
    return [x, -y, z]


def rotation_fru_to_flu(fru: list[3]):
    pitch = fru[0]
    roll = fru[1]
    yaw = fru[2]
    return [roll, -pitch, -yaw]


def matrix_fru_to_flu(T_fru):
    """
    将4x4变换矩阵从FRU坐标系转换到FLU坐标系

    参数:
    T_fru: 4x4 numpy数组, 在FRU坐标系中的变换矩阵

    返回:
    T_flu: 4x4 numpy数组, 在FLU坐标系中的变换矩阵
    """
    # 创建转换矩阵
    M_conv = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # 应用相似变换: T_flu = M_conv × T_fru × M_conv^(-1)
    # 由于M_conv是对角矩阵且M_conv = M_conv^(-1)
    T_flu = M_conv @ T_fru @ M_conv

    return T_flu


def polar_to_cartesian(altitude, azimuth, depth):
    """
    将雷达的极坐标(方位角、俯仰角、深度)转换为笛卡尔坐标(x, y, z)

    参数:
        altitude: 俯仰角(弧度)
        azimuth: 方位角(弧度)
        depth: 距离值(米)

    返回:
        (x, y, z) 笛卡尔坐标
    """
    # 计算坐标转换
    x = depth * np.cos(altitude) * np.cos(azimuth)
    y = depth * np.cos(altitude) * np.sin(azimuth)
    z = depth * np.sin(altitude)

    return x, y, z


def get_camera(file: Path, frame_id: str, stamp: int):
    msg = CompressedImage()
    msg.header.stamp.sec = int(stamp / 1e9)
    msg.header.stamp.nanosec = int(stamp % 1e9)
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    with open(file, "rb") as jpg_file:
        msg.data = jpg_file.read()

    return msg


def get_camera_info(sensors: dict, frame_id: str, stamp: int):
    msg = CameraInfo()
    msg.header.stamp.sec = int(stamp / 1e9)
    msg.header.stamp.nanosec = int(stamp % 1e9)
    msg.header.frame_id = frame_id
    msg.height = sensors[frame_id]["image_size_y"]
    msg.width = sensors[frame_id]["image_size_x"]
    msg.k[0:9] = [sensors[frame_id]["intrinsic"][r][c] for r in range(3) for c in range(3)]
    msg.r[0:9] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg.p[0:12] = [msg.k[0], msg.k[1], msg.k[2], 0, msg.k[3], msg.k[4], msg.k[5], 0, msg.k[6], msg.k[7], msg.k[8], 0]

    return msg


def get_lidar(file: Path, frame_id: str, stamp: int):
    las = laspy.read(file)
    x = las.x * 1.0 + 0.39
    y = las.y * -1.0
    z = las.z * 1.0 - 1.84
    points = np.vstack((x, y, z)).T
    # intensity = np.vstack((las.intensity)).T

    header = Header()
    header.stamp.sec = int(stamp / 1e9)
    header.stamp.nanosec = int(stamp % 1e9)
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    structured_data = np.array(
        [(points[i, 0], points[i, 1], points[i, 2], 100.0) for i in range(points.shape[0])],
        dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32)]),
    )

    return point_cloud2.create_cloud(header, fields, structured_data)


def get_radar(file: Path, dataset: str, frame_id: str, stamp: int):
    points = []

    with h5py.File(file, "r") as f:
        data_set = f[dataset]
        data = data_set[()]
        for detection in data:
            x, y, z = polar_to_cartesian(detection[1], detection[2], detection[0])
            points.append([x, y, z, detection[3]])

    header = Header()
    header.stamp.sec = int(stamp / 1e9)
    header.stamp.nanosec = int(stamp % 1e9)
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="velocity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    return point_cloud2.create_cloud(header, fields, points)


def get_tf(bounding_boxes: dict, stamp: int):
    msg = TFMessage()
    for bbox in bounding_boxes:
        if bbox["class"] == "ego_vehicle":
            tf = TransformStamped()
            tf.header.stamp.sec = int(stamp / 1e9)
            tf.header.stamp.nanosec = int(stamp % 1e9)
            tf.header.frame_id = "map"
            tf.child_frame_id = "base_link"
            translation = location_fru_to_flu(bbox["location"])
            tf.transform.translation.x = translation[0]
            tf.transform.translation.y = translation[1]
            tf.transform.translation.z = translation[2]
            rotation = rotation_fru_to_flu(bbox["rotation"])
            quaternion = R.from_euler("xyz", rotation, degrees=True).as_quat()
            tf.transform.rotation.w = quaternion[3]
            tf.transform.rotation.x = quaternion[0]
            tf.transform.rotation.y = quaternion[1]
            tf.transform.rotation.z = quaternion[2]
            msg.transforms.append(tf)

    return msg


def get_static_tf(sensors: dict, stamp: int):
    msg = TFMessage()
    for frame_id in sensors.keys():
        if "cam2ego" in sensors[frame_id]:
            T = np.array(sensors[frame_id]["cam2ego"])
            T_flu = matrix_fru_to_flu(T)
            translation = T_flu[0:3, 3]
            M_conv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            R_cam = T_flu[0:3, 0:3] @ M_conv
            quaternion = R.from_matrix(R_cam).as_quat()
        elif "lidar2ego" in sensors[frame_id]:
            T = np.array(sensors[frame_id]["lidar2ego"])
            T_flu = matrix_fru_to_flu(T)
            translation = T_flu[0:3, 3]
            quaternion = R.from_matrix(T_flu[0:3, 0:3]).as_quat()
        elif "radar2ego" in sensors[frame_id]:
            T = np.array(sensors[frame_id]["radar2ego"])
            T_flu = matrix_fru_to_flu(T)
            translation = T_flu[0:3, 3]
            quaternion = R.from_matrix(T_flu[0:3, 0:3]).as_quat()
        else:
            continue

        tf = TransformStamped()
        tf.header.stamp.sec = int(stamp / 1e9)
        tf.header.stamp.nanosec = int(stamp % 1e9)
        tf.header.frame_id = "base_link"
        tf.child_frame_id = frame_id
        tf.transform.translation.x = translation[0]
        tf.transform.translation.y = translation[1]
        tf.transform.translation.z = translation[2]
        tf.transform.rotation.w = quaternion[3]
        tf.transform.rotation.x = quaternion[0]
        tf.transform.rotation.y = quaternion[1]
        tf.transform.rotation.z = quaternion[2]
        msg.transforms.append(tf)

    return msg


def get_annotation_markers(bounding_boxes: dict, stamp: int):
    # for bbox in bounding_boxes:
    #     if bbox["class"] == "ego_vehicle":
    #         pass
    #     elif bbox["class"] == "vehicle":
    #         pass
    #     elif bbox["class"] == "traffic_light":
    #         pass
    #     elif bbox["class"] == "traffic_sign":
    #         pass
    #     elif bbox["class"] == "pedestrian":
    #         pass

    markers = MarkerArray()
    for bbox in bounding_boxes:
        marker = Marker()
        marker.header.stamp.sec = int(stamp / 1e9)
        marker.header.stamp.nanosec = int(stamp % 1e9)
        marker.header.frame_id = "map"
        marker.id = int(bbox["id"])
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        location = location_fru_to_flu(bbox["center"])
        marker.pose.position.x = location[0]
        marker.pose.position.y = location[1]
        marker.pose.position.z = location[2]
        rotation = rotation_fru_to_flu(bbox["rotation"])
        quaternion = R.from_euler("xyz", rotation, degrees=True).as_quat()
        marker.pose.orientation.w = quaternion[3]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.scale.x = bbox["extent"][0] * 2
        marker.scale.y = bbox["extent"][1] * 2
        marker.scale.z = bbox["extent"][2] * 2
        marker.lifetime.sec = 1
        # marker.text = ann["instance_token"]
        if bbox["class"] == "ego_vehicle":
            continue
            marker.ns = "ego_vehicle"
            marker.color.g = 1.0
            marker.color.a = 0.5
        elif bbox["class"] == "vehicle":
            marker.ns = "vehicle.car"
            marker.color.b = 1.0
            marker.color.a = 0.5
        elif bbox["class"] == "pedestrian":
            marker.ns = "human.pedestrian.adult"
            marker.color.r = 1.0
            marker.color.a = 0.5
        else:
            marker.ns = "movable_object.barrier"
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.5
        markers.markers.append(marker)
    return markers


def get_vector_map(map_explorer: NuScenesMapExplorer, bounding_boxes: dict, stamp: int):
    for bbox in bounding_boxes:
        if bbox["class"] == "ego_vehicle":
            translation = location_fru_to_flu(bbox["location"])
            rotation = rotation_fru_to_flu(bbox["rotation"])
            break
    patch_box = (translation[0], translation[1], 100, 100)
    yaw = rotation[2]
    msg = MarkerArray()

    # lane_divider = map_explorer._get_layer_line(patch_box, yaw, "lane_divider")
    # for i, line in enumerate(lane_divider):
    #     marker = Marker()
    #     marker.header.stamp.sec = int(stamp / 1e9)
    #     marker.header.stamp.nanosec = int(stamp % 1e9)
    #     marker.header.frame_id = "base_link"
    #     marker.ns = "lane_divider"
    #     marker.id = i
    #     marker.type = Marker.LINE_STRIP
    #     marker.action = Marker.ADD
    #     marker.pose.orientation.w = 1.0
    #     marker.scale.x = 0.2
    #     marker.color.r = 0.8
    #     marker.color.g = 0.8
    #     marker.color.b = 0.8
    #     marker.color.a = 1.0
    #     marker.lifetime.sec = 1
    #     for point in line.coords:
    #         point = Point(x=point[0], y=point[1], z=0.0)
    #         marker.points.append(point)
    #     msg.markers.append(marker)

    # road_divider = map_explorer._get_layer_line(patch_box, yaw, "road_divider")
    # for i, line in enumerate(road_divider):
    #     marker = Marker()
    #     marker.header.stamp.sec = int(stamp / 1e9)
    #     marker.header.stamp.nanosec = int(stamp % 1e9)
    #     marker.header.frame_id = "base_link"
    #     marker.ns = "road_divider"
    #     marker.id = i
    #     marker.type = Marker.LINE_STRIP
    #     marker.action = Marker.ADD
    #     marker.pose.orientation.w = 1.0
    #     marker.scale.x = 0.2
    #     marker.color.r = 0.8
    #     marker.color.g = 0.8
    #     marker.color.b = 0.8
    #     marker.color.a = 1.0
    #     marker.lifetime.sec = 1
    #     for point in line.coords:
    #         point = Point(x=point[0], y=point[1], z=0.0)
    #         marker.points.append(point)
    #     msg.markers.append(marker)

    ped_crossing = map_explorer._get_layer_polygon(patch_box, yaw, "ped_crossing")
    for i, multipolygon in enumerate(ped_crossing):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp.sec = int(stamp / 1e9)
            marker.header.stamp.nanosec = int(stamp % 1e9)
            marker.header.frame_id = "base_link"
            marker.ns = "ped_crossing"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 0.8
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    road_segment = map_explorer._get_layer_polygon(patch_box, yaw, "road_segment")
    for i, multipolygon in enumerate(road_segment):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp.sec = int(stamp / 1e9)
            marker.header.stamp.nanosec = int(stamp % 1e9)
            marker.header.frame_id = "base_link"
            marker.ns = "road_segment"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    lane = map_explorer._get_layer_polygon(patch_box, yaw, "lane")
    for i, multipolygon in enumerate(lane):
        for polygon in multipolygon.geoms:
            marker = Marker()
            marker.header.stamp.sec = int(stamp / 1e9)
            marker.header.stamp.nanosec = int(stamp % 1e9)
            marker.header.frame_id = "base_link"
            marker.ns = "lane"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.color.r = 0.6
            marker.color.g = 0.4
            marker.color.b = 0.8
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            for point in polygon.exterior.coords:
                point = Point(x=point[0], y=point[1], z=0.0)
                marker.points.append(point)
            msg.markers.append(marker)

    return msg


def write_scene_to_mcap(scene: Path, map: Path, rosbag: Path):
    timestamp_ns = time.time_ns()
    print(f"Using base timestamp {timestamp_ns}")
    scene_name = scene.name
    print(f"Loading scene {scene_name}")

    # Read nuScenes map
    map_name = scene_name.split("_")[1]
    print(f"Loading map {map_name}")
    nusc_map = NuScenesMap(dataroot=str(map), map_name=map_name)
    nusc_map_explorer = NuScenesMapExplorer(nusc_map)

    # Create writer
    rosbag.mkdir(parents=True, exist_ok=True)
    rosbag_path = rosbag.joinpath(scene_name)
    storage_options = rosbag2_py.StorageOptions(uri=str(rosbag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Create topics
    latch_qos = rosbag2_py._storage.QoS(1).transient_local()
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/map", "foxglove_msgs/msg/Grid", "cdr", [latch_qos]))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/semantic_map", "visualization_msgs/msg/MarkerArray", "cdr", [latch_qos]))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/imu", "sensor_msgs/msg/Imu", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/odom", "nav_msgs/msg/Odometry", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/pose", "geometry_msgs/msg/PoseWithCovarianceStamped", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/tf", "tf2_msgs/msg/TFMessage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/tf_static", "tf2_msgs/msg/TFMessage", "cdr", [latch_qos]))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/drivable_area", "foxglove_msgs/msg/Grid", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT_LEFT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_FRONT_RIGHT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_BACK_LEFT", "sensor_msgs/msg/PointCloud2", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/RADAR_BACK_RIGHT", "sensor_msgs/msg/PointCloud2", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/LIDAR_TOP", "sensor_msgs/msg/PointCloud2", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/compressed", "sensor_msgs/msg/CompressedImage", "cdr"))

    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/camera_info", "sensor_msgs/msg/CameraInfo", "cdr"))

    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/lidar", "foxglove_msgs/msg/ImageAnnotations", "cdr"))

    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_RIGHT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_RIGHT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_BACK_LEFT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/CAM_FRONT_LEFT/annotations", "foxglove_msgs/msg/ImageAnnotations", "cdr"))

    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/gps", "sensor_msgs/msg/NavSatFix", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/annotations", "visualization_msgs/msg/MarkerArray", "cdr"))
    # writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/car", "visualization_msgs/msg/MarkerArray", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, "/markers/map", "visualization_msgs/msg/MarkerArray", "cdr"))

    anno_path = scene.joinpath("anno")
    anno_files = [p for p in anno_path.rglob("*.json.gz") if p.is_file()]
    anno_files.sort()
    for index, file in enumerate(anno_files):
        cur_stamp = index * 100000000 + timestamp_ns
        with gzip.open(file, mode="rt", encoding="utf-8") as f:
            print(f"Reading {file} @ {cur_stamp}")

            anno = json.load(f)
            bounding_boxes = anno["bounding_boxes"]
            msg = get_annotation_markers(bounding_boxes, cur_stamp)
            writer.write("/markers/annotations", serialize_message(msg), cur_stamp)

            image_name = file.name.replace(".json.gz", ".jpg")
            msg = get_camera(scene.joinpath("camera", "rgb_front", image_name), "CAM_FRONT", cur_stamp)
            writer.write("/CAM_FRONT/compressed", serialize_message(msg), cur_stamp)
            msg = get_camera(scene.joinpath("camera", "rgb_front_right", image_name), "CAM_FRONT_RIGHT", cur_stamp)
            writer.write("/CAM_FRONT_RIGHT/compressed", serialize_message(msg), cur_stamp)
            msg = get_camera(scene.joinpath("camera", "rgb_back_right", image_name), "CAM_BACK_RIGHT", cur_stamp)
            writer.write("/CAM_BACK_RIGHT/compressed", serialize_message(msg), cur_stamp)
            msg = get_camera(scene.joinpath("camera", "rgb_back", image_name), "CAM_BACK", cur_stamp)
            writer.write("/CAM_BACK/compressed", serialize_message(msg), cur_stamp)
            msg = get_camera(scene.joinpath("camera", "rgb_back_left", image_name), "CAM_BACK_LEFT", cur_stamp)
            writer.write("/CAM_BACK_LEFT/compressed", serialize_message(msg), cur_stamp)
            msg = get_camera(scene.joinpath("camera", "rgb_front_left", image_name), "CAM_FRONT_LEFT", cur_stamp)
            writer.write("/CAM_FRONT_LEFT/compressed", serialize_message(msg), cur_stamp)

            sensors = anno["sensors"]
            msg = get_camera_info(sensors, "CAM_FRONT", cur_stamp)
            writer.write("/CAM_FRONT/camera_info", serialize_message(msg), cur_stamp)
            msg = get_camera_info(sensors, "CAM_FRONT_RIGHT", cur_stamp)
            writer.write("/CAM_FRONT_RIGHT/camera_info", serialize_message(msg), cur_stamp)
            msg = get_camera_info(sensors, "CAM_BACK_RIGHT", cur_stamp)
            writer.write("/CAM_BACK_RIGHT/camera_info", serialize_message(msg), cur_stamp)
            msg = get_camera_info(sensors, "CAM_BACK", cur_stamp)
            writer.write("/CAM_BACK/camera_info", serialize_message(msg), cur_stamp)
            msg = get_camera_info(sensors, "CAM_BACK_LEFT", cur_stamp)
            writer.write("/CAM_BACK_LEFT/camera_info", serialize_message(msg), cur_stamp)
            msg = get_camera_info(sensors, "CAM_FRONT_LEFT", cur_stamp)
            writer.write("/CAM_FRONT_LEFT/camera_info", serialize_message(msg), cur_stamp)

            msg = get_tf(bounding_boxes, cur_stamp)
            writer.write("/tf", serialize_message(msg), cur_stamp)
            msg = get_static_tf(sensors, cur_stamp)
            writer.write("/tf_static", serialize_message(msg), cur_stamp)

            pointcloud_name = file.name.replace(".json.gz", ".laz")
            msg = get_lidar(scene.joinpath("lidar", pointcloud_name), "LIDAR_TOP", cur_stamp)
            writer.write("/LIDAR_TOP", serialize_message(msg), cur_stamp)

            radar_name = file.name.replace(".json.gz", ".h5")
            msg = get_radar(scene.joinpath("radar", radar_name), "radar_front", "RADAR_FRONT", cur_stamp)
            writer.write("/RADAR_FRONT", serialize_message(msg), cur_stamp)
            msg = get_radar(scene.joinpath("radar", radar_name), "radar_front_left", "RADAR_FRONT_LEFT", cur_stamp)
            writer.write("/RADAR_FRONT_LEFT", serialize_message(msg), cur_stamp)
            msg = get_radar(scene.joinpath("radar", radar_name), "radar_front_right", "RADAR_FRONT_RIGHT", cur_stamp)
            writer.write("/RADAR_FRONT_RIGHT", serialize_message(msg), cur_stamp)
            msg = get_radar(scene.joinpath("radar", radar_name), "radar_back_left", "RADAR_BACK_LEFT", cur_stamp)
            writer.write("/RADAR_BACK_LEFT", serialize_message(msg), cur_stamp)
            msg = get_radar(scene.joinpath("radar", radar_name), "radar_back_right", "RADAR_BACK_RIGHT", cur_stamp)
            writer.write("/RADAR_BACK_RIGHT", serialize_message(msg), cur_stamp)

            msg = get_vector_map(nusc_map_explorer, bounding_boxes, cur_stamp)
            writer.write("/markers/map", serialize_message(msg), cur_stamp)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default=script_dir.joinpath("data"),
        help="path to bench2drive data directory",
    )
    parser.add_argument(
        "--map-dir",
        "-m",
        type=Path,
        default=script_dir.joinpath("map"),
        help="path to nuscenes data directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=script_dir.joinpath("output"),
        help="path to write rosbag into",
    )

    args = parser.parse_args()
    for scene in args.dataset_dir.iterdir():
        if scene.is_dir() and not scene.name.startswith("."):
            write_scene_to_mcap(scene, args.map_dir, args.output_dir)


if __name__ == "__main__":
    main()
