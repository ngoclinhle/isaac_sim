import os
import numpy as np
from sensor_msgs.msg import PointCloud2, Imu, PointField
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
import rosbag2_py

def TimeStamp(sec: float):
    msg = Time()
    msg.sec = int(sec)
    msg.nanosec = int((sec - msg.sec) * 1e9)
    return msg

class BagRecorder():
    def __init__(self, file_name):
        if not hasattr(self, 'writer'):
            self.writer = rosbag2_py.SequentialWriter()
            storage_options = rosbag2_py._storage.StorageOptions(
                uri=file_name,
                storage_id='sqlite3')
            converter_options = rosbag2_py._storage.ConverterOptions('', '')
            self.writer.open(storage_options, converter_options)
            # Create topics
            self.create_topics('/imu/data', 'sensor_msgs/msg/Imu')
            self.create_topics('/imu/pose', 'geometry_msgs/msg/PoseWithCovarianceStamped')
            self.create_topics('/tf', 'tf2_msgs/msg/TFMessage')
            self.create_topics('/points', 'sensor_msgs/msg/PointCloud2')
    
    def create_topics(self, topic_name, msg_type):
        topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic_name,
            type=msg_type,
            serialization_format='cdr')
        self.writer.create_topic(topic_info)
    
    def write_msg(self, topic_name, msg, time_sec):
        self.writer.write(
            topic_name,
            serialize_message(msg),
            int(time_sec*1e9))
    
    def write_imu(self, orientation, angular_velocity, linear_acceleration, frame_id, time_sec):
        # IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = TimeStamp(time_sec)
        imu_msg.header.frame_id = frame_id
        imu_msg.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
        imu_msg.angular_velocity = Vector3(x=angular_velocity[0], y=angular_velocity[1], z=angular_velocity[2])
        imu_msg.linear_acceleration = Vector3(x=linear_acceleration[0], y=linear_acceleration[1], z=linear_acceleration[2])
        
        # Example covariance, this should be adjusted based on your sensor accuracy
        imu_msg.orientation_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        imu_msg.angular_velocity_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        imu_msg.linear_acceleration_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        
        self.write_msg('/imu/data', imu_msg, time_sec)

        # Pose message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = TimeStamp(time_sec)
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
        
        # Fill a 6x6 covariance matrix for pose (orientation + position)
        pose_covariance = np.zeros((6, 6))
        
        # Set the orientation covariance (top-left 3x3 block)
        pose_covariance[0:3, 0:3] = np.reshape(imu_msg.orientation_covariance, (3, 3))
        
        # Optionally, you can set the position covariance (bottom-right 3x3 block)
        # Assuming we have some default or known covariance for position, which is zero here
        position_covariance = np.zeros((3, 3))
        pose_covariance[3:6, 3:6] = position_covariance
        
        # Flatten the 6x6 covariance matrix to fit the PoseWithCovariance format
        pose_msg.pose.covariance = pose_covariance.ravel().tolist()
        
        self.write_msg('/imu/pose', pose_msg, time_sec)

    def write_tf(self, translation, rotation, parent_frame, child_frame, time_sec):
        transform = TransformStamped()
        tf_msg = TFMessage()
        transform.header.stamp = TimeStamp(time_sec)
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
    
        tf_msg.transforms.append(transform)
        self.write_msg('/tf', tf_msg, time_sec)
    
    def write_pointcloud(self, points, frame_id, time_sec):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        header = Header(stamp=TimeStamp(time_sec), frame_id=frame_id)
        
        # Flatten the point cloud data
        point_data = np.column_stack((
            points['xyz'][..., 0].ravel(),  # x
            points['xyz'][..., 1].ravel(),  # y
            points['xyz'][..., 2].ravel(),  # z
            points['intensity'].ravel()     # intensity
        )).astype(np.float32).tobytes()
        
        if len(points['xyz'].shape) == 2:
            points['xyz'] = np.expand_dims(points['xyz'], axis=0)
        
        msg = PointCloud2(
            header=header,
            height=points['xyz'].shape[0],
            width=points['xyz'].shape[1],
            fields=fields,
            is_bigendian=False,
            point_step=16,
            row_step=16 * points['xyz'].shape[1],
            data=point_data,
            is_dense=True
        )
        self.write_msg('/points', msg, time_sec)

# Example to generate and record IMU, TF, and PointCloud2 data
def test_combined():
    bag_name = 'combined_bag_test'
    os.system('rm -rf ' + bag_name)
    rec = BagRecorder(bag_name)
    
    dt = 1.0/60  # time step
    steps = 60
    
    for i in range(steps):
        time_sec = i * dt
        
        # IMU Data
        orientation = [0.0, 0.0, np.sin(time_sec/2), np.cos(time_sec/2)]  # Quaternion (x, y, z, w)
        angular_velocity = [0.1*time_sec, 0.2*time_sec, 0.3*time_sec]  # rad/s
        linear_acceleration = [0.1, 0.0, 9.81]  # m/s^2
        rec.write_imu(orientation, angular_velocity, linear_acceleration, 'base_link', time_sec)
        
        # TF Data
        translation = [0.1*time_sec, 0.2*time_sec, 0.3*time_sec]
        rotation = [0.0, 0.0, np.sin(time_sec/2), np.cos(time_sec/2)]
        rec.write_tf(translation, rotation, 'world', 'base_link', time_sec)
        
        # PointCloud2 Data
        w = 100
        h = 32
        r = 1.0
        theta = np.linspace(0, 2*np.pi, w)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.linspace(-1, 1, h)
        xx, zz = np.meshgrid(x, z)
        yy = np.tile(y, (h, 1))
        xyz = np.stack([xx, yy, zz], axis=-1)
        intensity = np.ones_like(xyz[..., 0])
        rec.write_pointcloud({'xyz': xyz, 'intensity': intensity}, 'base_link', time_sec)
    
    del rec

if __name__ == "__main__":
    test_combined()
