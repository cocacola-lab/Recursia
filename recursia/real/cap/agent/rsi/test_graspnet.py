"""Model frame, TCP depth, jaw width and optical point-cloud contract tests."""
import io
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .graspnet import ContactGraspNetClient, encode_array, decode_array, select_grasp, default_grasp_to_ee


def test_contact_frame_to_franka_tcp_not_camera_or_world_offset():
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("y", .3).as_matrix()
    camera[:3, 3] = [1., .1, .4]
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    grasp[:3, 3] = [.1, .2, .8]
    result, width, index = select_grasp(grasp[None], [.9], [.04], camera, default_grasp_to_ee())
    expected_contact = camera @ grasp @ np.array([0, 0, .1034, 1])
    np.testing.assert_allclose(result[:3, 3], expected_contact[:3])
    # EE y is the negative network x; approach (z) stays unchanged.
    np.testing.assert_allclose(result[:3, 1], -(camera @ grasp)[:3, 0], atol=1e-12)
    np.testing.assert_allclose(result[:3, 2], (camera @ grasp)[:3, 2])
    assert width == .04 and index == 0


def test_score_selects_learned_candidate_with_physical_hand_width():
    grasps = np.tile(np.eye(4), (3, 1, 1))
    result, width, index = select_grasp(grasps, [.99, .8, .9], [.09, .04, .07], np.eye(4), np.eye(4))
    assert index == 2 and width == .07


@pytest.mark.parametrize("grasps,scores,widths,match", [
    (np.empty((0,4,4)), [], [], "no grasps"),
    (np.eye(4)[None], [.9], [.09], "hand opening"),
    (np.eye(4)[None], [.9], [-.004], "hand opening"),
    (np.eye(4)[None], [np.nan], [.04], "non-finite"),
    (np.eye(4)[None], [.9,.8], [.04], "shapes"),
])
def test_invalid_network_candidates_are_not_replaced_by_geometric_guess(grasps, scores, widths, match):
    with pytest.raises(ValueError, match=match):
        select_grasp(grasps, scores, widths, np.eye(4), np.eye(4))


def test_thin_cup_rim_is_not_filtered_by_old_five_mm_minimum():
    _, width, index = select_grasp(np.eye(4)[None], [.9], [.0039721765], np.eye(4), np.eye(4))
    assert width == .0039721765 and index == 0


def test_hand_measurement_sets_maximum_instead_of_old_75_mm_limit():
    grasps = np.tile(np.eye(4), (2, 1, 1))
    _, width, index = select_grasp(grasps, [.95, .8], [.078, .074], np.eye(4), np.eye(4), max_width_m=.08)
    assert index == 0 and width == .078
    _, width, index = select_grasp(grasps, [.95, .8], [.078, .074], np.eye(4), np.eye(4), max_width_m=.076)
    assert index == 1 and width == .074


def test_client_sends_optical_points_and_retains_predicted_width(monkeypatch):
    from cap.saved_scripts.fr3_runtime.camera import Capture
    capture = Capture(np.zeros((20,20,3), np.uint8), np.full((20,20), .8),
                      np.array([[100,0,10],[0,100,10],[0,0,1]]), "test", "test")
    mask = np.ones((20,20), bool)
    def response(request, timeout):
        assert request.full_url.endswith("/plan_point_clouds")
        data = json.loads(request.data)
        points = decode_array(data["pc_segment_base64"])
        np.testing.assert_allclose(points[:,2], .8)
        assert np.max(np.abs(points[:,:2])) < .09  # Not transformed into base frame.
        return io.StringIO(json.dumps({"model":"contact_graspnet_pytorch", "frame":"camera_optical",
            **{key+"_base64":encode_array(value) for key,value in {
                "grasps":np.eye(4)[None], "scores":[.9], "widths":[.041], "contact_pts":[[0,0,.8]]}.items()}}))
    monkeypatch.setattr("urllib.request.urlopen", response)
    result = ContactGraspNetClient().predict(capture, mask)
    assert result["widths"][0] == .041
