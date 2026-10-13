# Franka FR3 model

This is the FR3 MJCF model used by the robot runtime for forward kinematics,
inverse kinematics, joint limits, and collision previews. The XML and its 36
referenced meshes are bundled without changes. The model represents the arm;
it does not include the hand or other scene objects.

The model originates from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
FR3 description, converted from the
[Franka robot description](https://github.com/frankaemika/franka_description).
It requires MuJoCo 3.1.3 or later and is distributed under the
[Apache License 2.0](LICENSE). See the XML and mesh files for retained notices.
