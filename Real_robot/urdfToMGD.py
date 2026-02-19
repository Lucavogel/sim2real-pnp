#!/usr/bin/env python3
"""
MGD / Forward Kinematics depuis un URDF (sans dépendances ROS)

Support:
- joints: fixed, revolute, continuous, prismatic
- origin xyz + rpy
- axis
- chaîne base_link -> ee_link

Usage:
  - Mets URDF_PATH, BASE_LINK, EE_LINK
  - Mets Q_MAP avec les valeurs des joints (radians / mètres)
  - Lance: python3 fk_from_urdf.py
"""

import math
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


# ---------------------------- Math utils ----------------------------

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from RPY (URDF convention: fixed axis rotations Rz(yaw)*Ry(pitch)*Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def T_from_R_p(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def T_from_xyz_rpy(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    R = rpy_to_R(rpy[0], rpy[1], rpy[2])
    p = np.array(xyz, dtype=np.float64)
    return T_from_R_p(R, p)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.tolist()
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def R_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues."""
    axis = axis.astype(np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = axis / n
    K = skew(a)
    I = np.eye(3, dtype=np.float64)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


# ---------------------------- URDF structures ----------------------------

@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_T: np.ndarray          # parent_link -> joint frame (fixed)
    axis: np.ndarray              # joint axis in joint frame
    limit: Optional[Tuple[float, float]] = None


def parse_xyz(attr: Optional[str]) -> Tuple[float, float, float]:
    if not attr:
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in attr.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Bad xyz: {attr}")
    return (vals[0], vals[1], vals[2])


def parse_rpy(attr: Optional[str]) -> Tuple[float, float, float]:
    if not attr:
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in attr.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Bad rpy: {attr}")
    return (vals[0], vals[1], vals[2])


def load_urdf_joints(urdf_path: str) -> Dict[str, Joint]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints: Dict[str, Joint] = {}

    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]

        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            raise ValueError(f"Joint {name} missing parent/child")

        parent = parent_el.attrib["link"]
        child = child_el.attrib["link"]

        origin_el = j.find("origin")
        xyz = parse_xyz(origin_el.attrib.get("xyz") if origin_el is not None else None)
        rpy = parse_rpy(origin_el.attrib.get("rpy") if origin_el is not None else None)
        origin_T = T_from_xyz_rpy(xyz, rpy)

        axis_el = j.find("axis")
        if axis_el is not None and "xyz" in axis_el.attrib:
            ax = np.array(parse_xyz(axis_el.attrib["xyz"]), dtype=np.float64)
        else:
            ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # URDF default

        limit_el = j.find("limit")
        lim = None
        if limit_el is not None and "lower" in limit_el.attrib and "upper" in limit_el.attrib:
            lim = (float(limit_el.attrib["lower"]), float(limit_el.attrib["upper"]))

        joints[name] = Joint(
            name=name,
            jtype=jtype,
            parent=parent,
            child=child,
            origin_T=origin_T,
            axis=ax,
            limit=lim,
        )

    return joints


def build_parent_map(joints: Dict[str, Joint]) -> Dict[str, Joint]:
    """Map child_link -> Joint that connects parent->child."""
    child_to_joint: Dict[str, Joint] = {}
    for jt in joints.values():
        child_to_joint[jt.child] = jt
    return child_to_joint


def chain_links(child_to_joint: Dict[str, Joint], base_link: str, ee_link: str) -> List[Joint]:
    """Return ordered joints from base_link to ee_link."""
    chain: List[Joint] = []
    cur = ee_link

    # Walk backwards from ee to base
    while cur != base_link:
        if cur not in child_to_joint:
            raise ValueError(f"No joint found that leads to link '{cur}'. Check base/ee names.")
        jt = child_to_joint[cur]
        chain.append(jt)
        cur = jt.parent

    chain.reverse()
    return chain


def joint_motion_T(joint: Joint, q: float) -> np.ndarray:
    """
    Transform from joint frame to child link frame induced by joint variable q.
    In URDF, joint axis is expressed in the joint frame after origin.
    """
    if joint.jtype in ("fixed",):
        return np.eye(4, dtype=np.float64)

    axis = joint.axis.astype(np.float64)
    if joint.jtype in ("revolute", "continuous"):
        R = R_from_axis_angle(axis, q)
        return T_from_R_p(R, np.zeros(3, dtype=np.float64))

    if joint.jtype in ("prismatic",):
        n = np.linalg.norm(axis)
        a = axis / (n if n > 1e-12 else 1.0)
        p = a * q
        return T_from_R_p(np.eye(3, dtype=np.float64), p)

    raise ValueError(f"Unsupported joint type: {joint.jtype}")


def forward_kinematics_urdf(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    q_map: Dict[str, float],
    return_all: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute FK from base_link to ee_link.

    q_map: dict {joint_name: q_value} in radians (revolute/continuous), meters (prismatic).
    return_all: if True return transforms for all links on chain (base->...->ee)
    """
    joints = load_urdf_joints(urdf_path)
    child_to_joint = build_parent_map(joints)
    chain = chain_links(child_to_joint, base_link, ee_link)

    T = np.eye(4, dtype=np.float64)

    link_T: Dict[str, np.ndarray] = {base_link: T.copy()}
    cur_link = base_link

    for jt in chain:
        if jt.parent != cur_link:
            # Should never happen if chain is correct, but guard
            raise RuntimeError(f"Chain mismatch: expected parent {cur_link}, got {jt.parent}")

        q = float(q_map.get(jt.name, 0.0))  # default 0 if not provided
        T = T @ jt.origin_T @ joint_motion_T(jt, q)

        cur_link = jt.child
        if return_all:
            link_T[cur_link] = T.copy()

    return T, link_T


# ---------------------------- Example usage ----------------------------

if __name__ == "__main__":
    # ✅ Mets ton URDF ici
    URDF_PATH = "/home/ajin/workspace/sim2real-pnp/environ/ur10/ur10.urdf"

    # ✅ Mets les bons noms de liens (regarde dans l'URDF: <link name="...">)
    BASE_LINK = "base_link"
    EE_LINK = "tool0"

    # ✅ Donne tes q (radians). Les joints non présents => 0.0
    # Exemple UR10:
    Q_MAP = {
        "shoulder_pan_joint": 0.08866273,
        "shoulder_lift_joint": -1.6421803,
        "elbow_joint": 1.5838863,
        "wrist_1_joint": 0.05846853,
        "wrist_2_joint": -3.21141,
        "wrist_3_joint": 0.0,
    }

    T_be, link_T = forward_kinematics_urdf(
        URDF_PATH,
        BASE_LINK,
        EE_LINK,
        Q_MAP,
        return_all=True,
    )

    pos = T_be[:3, 3]
    R = T_be[:3, :3]

    np.set_printoptions(precision=6, suppress=True)
    print("\n==================== FK RESULT ====================")
    print(f"URDF: {URDF_PATH}")
    print(f"Base: {BASE_LINK}  ->  EE: {EE_LINK}")
    print("Position (m):", pos)
    print("Rotation R:\n", R)
    print("T_base_ee:\n", T_be)

    # Optionnel: afficher les liens traversés
    print("\nLinks on chain:")
    for k in link_T.keys():
        print(" -", k)
