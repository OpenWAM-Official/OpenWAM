"""Builds task_specs.json: one ordered stage chain per benchmark scene for verify.py.

Stages are sim checks (see verify.CHECKS) or {"check": "judge"} when only a vision judge can decide the last step
(no liquid / no pudding / no button in the sim). Object names come from the scene layouts; thresholds are first
guesses to be calibrated on data/webxr_episodes (calibrate.py). Keyed by 5-digit scene prefix here, expanded to
the full scene id from tasks.json on write.

run: python benchmarks/sharpa_webxr/_build_task_specs.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = json.loads((HERE / "tasks.json").read_text())
CATS = json.loads((HERE / "task_categories.json").read_text())

PAN_DISH = [[-0.09, -0.109, -0.016], [0.09, 0.07, 0.022]]  # world_pan_01 without its +y handle


def reach(o):
    return {"name": "reach", "check": "reach", "obj": o}


def grasp(o, **kw):
    return {"name": "grasp", "check": "grasp", "obj": o, **kw}


def pick(o, lift=0.05, hold=1.0, final=False):
    # "pick up X" is labelled on ever having held it; "grasp/grip X firmly" (final=True) also requires it was not
    # dropped afterwards (still held at the end, or set down gently)
    return {"name": "lift_hold", "check": "hold_lifted", "obj": o, "lift_m": lift, "hold_s": hold, **({"end_rule": "not_dropped"} if final else {})}


def place(o, t, **kw):
    return {"name": "place", "check": "place_in", "obj": o, "target": t, **kw}


def judge(what):
    return {"name": "complete", "check": "judge", "question": what}


S = {
    # ---- pick and place ------------------------------------------------------------------------------------
    "00001": ["world_butter", "world_pan_01", [reach("world_butter"), grasp("world_butter"), place("world_butter", "world_pan_01", target_box=PAN_DISH, contact="world_pan_01")]],
    "00019": ["world_bread_02", "world_bread_03", [reach("world_bread_02"), grasp("world_bread_02"), place("world_bread_02", "world_bread_03", contact="world_bread_03", xy_margin_m=-0.01)]],
    "00286": ["world_bread_03", "world_plate", [reach("world_bread_03"), grasp("world_bread_03"), place("world_bread_03", "world_plate", contact="world_plate")]],
    "00381": ["world_flower_01", None, [reach("world_flower_01"), grasp("world_flower_01"), {"name": "out_of_box", "check": "move_out", "obj": "world_flower_01", "min_disp_m": 0.12, "release": False, "rest": False, "final": True}]],
    "00391": ["world_plate_04", None, [reach("world_plate_04"), grasp("world_plate_04"), {"name": "out_of_oven", "check": "move_out", "obj": "world_plate_04", "min_disp_m": 0.15, "release": False, "rest": False, "final": True}]],
    "00457": ["world_test_tube_01", "world_rack", [reach("world_test_tube_01"), grasp("world_test_tube_01"), place("world_test_tube_01", "world_rack", z_above_m=0.12, contact="world_rack")]],
    "00636": ["world_teapot_lid", None, [reach("world_teapot_lid"), grasp("world_teapot_lid"), {"name": "lid_off_and_down", "check": "move_out", "obj": "world_teapot_lid", "min_disp_m": 0.10}]],
    "00646": ["world_phone_01", "world_phone_case", [reach("world_phone_01"), grasp("world_phone_01"), {"name": "out_of_case", "check": "move_out", "obj": "world_phone_01", "target": "world_phone_case", "min_disp_m": 0.08, "release": False, "rest": False, "final": True}]],
    "00807": ["world_egg", "world_pan_01", [reach("world_egg"), grasp("world_egg"), place("world_egg", "world_pan_01", target_box=PAN_DISH, contact="world_pan_01")]],
    "00921": ["world_butter", "world_pan", [reach("world_butter"), grasp("world_butter"), place("world_butter", "world_pan", xy_margin_m=0.03, contact="world_pan")]],
    "01012": ["world_eyeshadow_brush", "world_eyeshadow_box", [reach("world_eyeshadow_brush"), grasp("world_eyeshadow_brush"), place("world_eyeshadow_brush", "world_eyeshadow_box", z_above_m=0.04, contact="world_eyeshadow_box")]],
    "01083": ["world_coin", "world_game_box", [reach("world_coin"), grasp("world_coin"), place("world_coin", "world_game_box", z_above_m=0.03, contact="world_game_box")]],
    "01085": ["world_coin", "world_game_box", [reach("world_coin"), grasp("world_coin"), place("world_coin", "world_game_box", z_above_m=0.03, contact="world_game_box")]],
    "01089": ["world_dice", "world_game_board", [reach("world_dice"), grasp("world_dice"), place("world_dice", "world_game_board", contact="world_game_board")]],
    "01242": ["world_bowl", None, [judge("banana pieces transferred into the salad bowl (no banana bodies in the sim)")]],
    "01440": ["world_shovel", "world_pot", [reach("world_shovel"), grasp("world_shovel"), place("world_shovel", "world_pot", z_above_m=0.12, contact="world_pot")]],
    "self-": ["synthesis_cup_cup_005_5e43862263a34727", None, [reach("synthesis_cup_cup_005_5e43862263a34727"), grasp("synthesis_cup_cup_005_5e43862263a34727"), judge("cup placed on the cup rack (rack not simulated)")]],
    # ---- articulated open/close ----------------------------------------------------------------------------
    "00005": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "adjusted", "check": "joint_goal", "obj": "world_laptop_03", "delta": -0.15, "min": -0.6, "max": 0.12, "release": True, "hold_s": 1.0, "final": True}]],
    "00079": ["world_gate_01", None, [reach("world_gate_01"), {"name": "closed", "check": "joint_goal", "obj": "world_gate_01", "joint": "articraft_rec_microwave_oven_0002_joints_root_to_door", "max": 0.06, "hold_s": 0.5, "final": True}]],
    "00082": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "closed", "check": "joint_goal", "obj": "world_laptop_03", "min": 1.85, "hold_s": 0.5, "max_speed": 4.6, "final": True}]],
    "00337": ["world_laptop_03", None, [reach("world_laptop_03"), {"name": "opened", "check": "joint_goal", "obj": "world_laptop_03", "max": 1.25, "hold_s": 1.0, "final": True}]],
    "00626": ["world_rec_desk_with_drawer_0005", None, [reach("world_rec_desk_with_drawer_0005"), {"name": "pulled_out", "check": "joint_goal", "obj": "world_rec_desk_with_drawer_0005", "min": 0.08, "hold_s": 0.5, "final": True}]],
    "01000": ["world_burger_box", None, [reach("world_burger_box"), {"name": "opened", "check": "joint_goal", "obj": "world_burger_box", "delta": -0.9, "hold_s": 0.5, "final": True}]],
    "01018": ["world_box", None, [reach("world_box"), {"name": "closed", "check": "joint_goal", "obj": "world_box", "max": 0.15, "hold_s": 0.3},
                                 {"name": "faces_robot", "check": "orientation", "obj": "world_box", "mode": "face", "axis": [1, 0, 0], "toward": [-0.035, 0.999], "tol_deg": 15, "keep_upright_deg": 30, "hold_s": 0.5, "final": True}]],
    "01747": ["world_pencil_case", None, [reach("world_pencil_case"), {"name": "opened", "check": "joint_goal", "obj": "world_pencil_case", "max": -1.0, "hold_s": 0.5, "final": True}]],
    # ---- cap / lid ---------------------------------------------------------------------------------------
    "00023": ["world_lid_09", "world_box", [reach("world_lid_09"), grasp("world_lid_09"), {"name": "lid_on", "check": "relative_pose", "obj": "world_lid_09", "ref": "world_box", "point": [0, 0, 0.004], "xy_tol_m": 0.03, "z_tol_m": 0.02, "z_above_m": 0.03, "tilt_deg": 12, "contact": "world_box"}]],
    "00075": ["world_pen_cap", "world_pen", [reach("world_pen_cap"), grasp("world_pen_cap"), {"name": "capped", "check": "relative_pose", "obj": "world_pen_cap", "ref": "world_pen", "point": [0.038, 0, 0.016], "offset": [0.041, 0.002, 0.012], "pos_tol_m": 0.012, "tilt_deg": 90, "contact": "world_pen", "release": False, "rest": False}]],
    "00077": ["world_teapot_lid", "world_teapot", [reach("world_teapot_lid"), grasp("world_teapot_lid"), {"name": "lid_on", "check": "relative_pose", "obj": "world_teapot_lid", "ref": "world_teapot", "xy_tol_m": 0.035, "z_tol_m": 0.04, "z_above_m": 0.05, "tilt_deg": 15, "contact": "world_teapot"}]],
    "00894": ["world_cap_07", "world_body_03", [reach("world_cap_07"), grasp("world_cap_07"), {"name": "capped", "check": "relative_pose", "obj": "world_cap_07", "ref": "world_body_03", "point": [0, 0, 0.083], "xy_tol_m": 0.03, "z_tol_m": 0.03, "z_above_m": 0.04, "tilt_deg": 15, "contact": "world_body_03"}]],
    # ---- pick / grasp only --------------------------------------------------------------------------------
    "00177": ["world_ukelele_01", None, [reach("world_ukelele_01"), pick("world_ukelele_01", final=True)]],
    "00178": ["world_sauce_01", None, [reach("world_sauce_01"), pick("world_sauce_01", final=True)]],
    "00375": ["world_bag_02", None, [reach("world_bag_02"), pick("world_bag_02")]],
    "01116": ["world_drill", None, [reach("world_drill"), pick("world_drill", final=True)]],
    "01174": ["world_stamp", None, [reach("world_stamp"), pick("world_stamp")]],
    "01304": ["world_kettle", None, [reach("world_kettle"), pick("world_kettle")]],
    "01337": ["world_scissors", None, [reach("world_scissors"), pick("world_scissors")]],
    # ---- reorient / flip / rotate -------------------------------------------------------------------------
    "01032": ["world_box", None, [reach("world_box"), {"name": "upturned", "check": "orientation", "obj": "world_box", "mode": "flip", "tol_deg": 35, "final": True, "release": False, "rest": False}]],
    "01041": ["world_game_board", None, [reach("world_game_board"), {"name": "flipped", "check": "orientation", "obj": "world_game_board", "mode": "flip", "tol_deg": 35, "final": True, "release": False, "rest": False}]],
    "01246": ["world_bowl", None, [reach("world_bowl"), {"name": "upturned", "check": "orientation", "obj": "world_bowl", "mode": "flip", "tol_deg": 45, "final": True, "release": False, "rest": False}]],
    "01541": ["world_box", None, [reach("world_box"), grasp("world_box", lift_m=0.02, carry_m=0.03), {"name": "rotated_side", "check": "orientation", "obj": "world_box", "mode": "rotate", "min_deg": 60}]],
    "01542": ["world_teacup", None, [reach("world_teacup"), {"name": "handle_faces_robot", "check": "orientation", "obj": "world_teacup", "mode": "face", "axis": [1, 0, 0], "toward": [1, 0], "tol_deg": 25, "keep_upright_deg": 30, "final": True}]],
    "01564": ["world_tablet", None, [reach("world_tablet"), {"name": "portrait", "check": "orientation", "obj": "world_tablet", "mode": "rotate", "min_deg": 55, "final": True, "release": False, "rest": False}]],
    # ---- shake / toss -------------------------------------------------------------------------------------
    "00779": ["world_flask_02", None, [reach("world_flask_02"), grasp("world_flask_02"), {"name": "shaken", "check": "oscillation", "obj": "world_flask_02", "min_reversals": 4, "min_amp_m": 0.03}]],
    "01592": ["world_dice", None, [reach("world_dice"), grasp("world_dice", lift_m=0.02), {"name": "shaken", "check": "oscillation", "obj": "world_dice", "min_reversals": 4, "min_amp_m": 0.02}]],
    "01605": ["world_nail_polish_bottle", None, [reach("world_nail_polish_bottle"), grasp("world_nail_polish_bottle"), {"name": "shaken", "check": "oscillation", "obj": "world_nail_polish_bottle", "min_reversals": 4, "min_amp_m": 0.03}]],
    "01652": ["world_coin", None, [reach("world_coin"), grasp("world_coin", lift_m=0.02), {"name": "tossed_caught", "check": "free_flight", "obj": "world_coin", "min_rise_m": 0.04, "min_rel_rise_m": 0.02, "catch_dist_m": 0.22}]],
    # ---- pour (sim: grasp, carry over, tilt; judge: the pour itself) ----------------------------------------
    "00175": ["world_kettle", "world_mug_025", [reach("world_kettle"), grasp("world_kettle"), {"name": "tilt_over_target", "check": "tilt_over", "obj": "world_kettle", "target": "world_mug_025", "min_tilt_deg": 30, "xy_slack_m": 0.12}, judge("water poured from the kettle into the mug")]],
    "00475": ["world_beaker_01", "world_flask_02", [reach("world_beaker_01"), grasp("world_beaker_01"), {"name": "tilt_over_target", "check": "tilt_over", "obj": "world_beaker_01", "target": "world_flask_02", "min_tilt_deg": 50, "xy_slack_m": 0.08}, judge("substance poured from the beaker into the conical flask")]],
    "00487": ["world_body_04", "world_bowl", [reach("world_body_04"), grasp("world_body_04"), {"name": "tilt_over_target", "check": "tilt_over", "obj": "world_body_04", "target": "world_bowl", "min_tilt_deg": 50, "xy_slack_m": 0.08}, judge("tea poured from the bottle into the bowl")]],
    "01480": ["world_bowl", "world_pan", [reach("world_bowl"), grasp("world_bowl"), {"name": "tilt_over_target", "check": "tilt_over", "obj": "world_bowl", "target": "world_pan", "min_tilt_deg": 50, "xy_slack_m": 0.05}, judge("bowl contents poured into the pan")]],
    "01498": ["world_kettle", "world_teapot", [reach("world_kettle"), grasp("world_kettle"), {"name": "tilt_over_target", "check": "tilt_over", "obj": "world_kettle", "target": "world_teapot", "min_tilt_deg": 30, "xy_slack_m": 0.12}, judge("hot water poured from the kettle into the teapot")]],
    # ---- tool use (sim: grasp tool, tool acts on target; judge: the effect) ---------------------------------
    "00051": ["world_eraser_07", "world_plate_05", [reach("world_eraser_07"), grasp("world_eraser_07", lift_m=0.01, carry_m=0.03), {"name": "brushed", "check": "sweep", "obj": "world_eraser_07", "target": "world_plate_05", "contact": True, "min_path_m": 0.15}, judge("plate brushed with the eraser")]],
    "00749": ["world_spoon_08", "world_pan_06", [reach("world_spoon_08"), grasp("world_spoon_08"), {"name": "stirred", "check": "sweep", "obj": "world_spoon_08", "target": "world_pan_06", "xy_slack_m": 0.03, "z_above_m": 0.12, "min_path_m": 0.30}, judge("pot stirred with the spoon")]],
    "00787": ["world_spatula", "world_pan_01", [reach("world_spatula"), grasp("world_spatula"), {"name": "scrambled", "check": "sweep", "obj": "world_spatula", "target": "world_pan_01", "target_box": [[-0.076, -0.097, -0.019], [0.072, 0.09, 0.042]], "xy_slack_m": 0.03, "z_above_m": 0.10, "min_path_m": 0.30}, judge("egg scrambled by moving the spatula around the pan")]],
    "00863": ["world_hammer_03", "world_box_13", [reach("world_hammer_03"), grasp("world_hammer_03"), {"name": "hit", "check": "contact_events", "obj": "world_hammer_03", "target": "world_box_13", "min_events": 1, "min_speed_m_s": 0.15}, judge("box hit with the hammer head")]],
    "01477": ["world_camera", None, [reach("world_camera"), pick("world_camera", lift=0.03, hold=0.5, final=False), judge("camera button clicked (no button in the sim)")]],
    "01587": ["world_spoon", "world_plate", [reach("world_spoon"), grasp("world_spoon"), {"name": "scooped", "check": "sweep", "obj": "world_spoon", "target": "world_plate", "xy_slack_m": 0.05, "z_above_m": 0.08, "min_path_m": 0.05}, judge("a piece of pudding scooped from the plate with the spoon")]],
    "01640": ["world_drumstick", "world_drum", [reach("world_drumstick"), grasp("world_drumstick"), {"name": "hit", "check": "contact_events", "obj": "world_drumstick", "target": "world_drum", "min_events": 1, "min_speed_m_s": 0.15}, judge("drumhead hit with the drum stick")]],
}


def main():
    full = {sid[:5]: sid for sid in TASKS}
    assert set(S) == set(full), (set(S) ^ set(full))
    out = {}
    for short, (obj, target, stages) in S.items():
        sid = full[short]
        out[sid] = {"version": 1, "category": CATS[sid], "task": TASKS[sid]["task"], "task_object": obj, "target": target,
                    "no_fall": [o for o in (obj, target) if o], "stages": stages,
                    "needs_judge": any(s["check"] == "judge" for s in stages)}
    (HERE / "task_specs.json").write_text(json.dumps(out, indent=1))
    n_j = sum(v["needs_judge"] for v in out.values())
    print(f"wrote {len(out)} specs, {n_j} end in a judge stage")


if __name__ == "__main__":
    main()
