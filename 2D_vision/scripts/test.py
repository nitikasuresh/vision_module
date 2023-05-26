from autolab_core import RigidTransform, Point


realsense_to_ee_transform = RigidTransform.load("calib/realsense_ee.tf")
print("\nTransform: ", realsense_to_ee_transform)