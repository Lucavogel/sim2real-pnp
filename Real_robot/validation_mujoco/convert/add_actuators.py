#!/usr/bin/env python3
"""
Ajoute les actuateurs, lumières et options au XML MuJoCo généré.
"""
import xml.etree.ElementTree as ET
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

INPUT_XML = "ur10.xml"
OUTPUT_XML = "ur10_complete.xml"

print(f"⚙️  Ajout des actuateurs et configuration dans {INPUT_XML}")

tree = ET.parse(INPUT_XML)
root = tree.getroot()

# 1. Options (timestep, gravity)
option = root.find('option')
if option is None:
    option = ET.SubElement(root, 'option')
option.set('timestep', '0.001')
option.set('gravity', '0 0 -9.81')
print("  ✓ Options ajoutées")

# 2. Lumières au worldbody
worldbody = root.find('worldbody')
lights_exist = any(child.tag == 'light' for child in worldbody)
if not lights_exist:
    ET.SubElement(worldbody, 'light', {
        'pos': '0 0 3',
        'dir': '0 0 -1',
        'diffuse': '0.9 0.9 0.9'
    })
    ET.SubElement(worldbody, 'light', {
        'pos': '1 1 3',
        'dir': '-1 -1 -1',
        'diffuse': '0.6 0.6 0.6'
    })
    print("  ✓ Lumières ajoutées")

# 3. Sol avec texture
asset = root.find('asset')
if asset is None:
    asset = ET.Element('asset')
    # Insérer après <compiler> s'il existe
    compiler = root.find('compiler')
    if compiler is not None:
        root.insert(list(root).index(compiler) + 1, asset)
    else:
        root.insert(0, asset)

grid_tex = any(child.get('name') == 'grid' for child in asset.iter('texture'))
if not grid_tex:
    ET.SubElement(asset, 'texture', {
        'name': 'grid',
        'type': '2d',
        'builtin': 'checker',
        'width': '512',
        'height': '512',
        'rgb1': '.1 .2 .3',
        'rgb2': '.2 .3 .4'
    })
    ET.SubElement(asset, 'material', {
        'name': 'grid',
        'texture': 'grid',
        'texrepeat': '1 1',
        'texuniform': 'true',
        'reflectance': '.2'
    })
    print("  ✓ Texture sol ajoutée")

floor_exists = any(child.get('name') == 'floor' for child in worldbody.iter('geom'))
if not floor_exists:
    ET.SubElement(worldbody, 'geom', {
        'name': 'floor',
        'type': 'plane',
        'size': '2 2 0.01',
        'material': 'grid'
    })
    print("  ✓ Sol ajouté")

# 4. Actuateurs (les plus importants !)
actuator = root.find('actuator')
if actuator is None or len(actuator) == 0:
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')
    
    joints = [
        ('shoulder_pan_joint', 10000, 1000),
        ('shoulder_lift_joint', 10000, 1000),
        ('elbow_joint', 10000, 1000),
        ('wrist_1_joint', 5000, 500),
        ('wrist_2_joint', 5000, 500),
        ('wrist_3_joint', 5000, 500),
    ]
    
    for joint_name, kp, kv in joints:
        ET.SubElement(actuator, 'position', {
            'name': joint_name.replace('_joint', ''),
            'joint': joint_name,
            'kp': str(kp),
            'kv': str(kv)
        })
    
    print(f"  ✓ {len(joints)} actuateurs ajoutés")

# Formater proprement
ET.indent(tree, space='  ')
tree.write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)

print(f"\n✅ XML complet sauvegardé: {OUTPUT_XML}")
print("👉 Utilise ce fichier dans ton script de test MuJoCo!")
