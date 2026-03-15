import json
from pymol import cmd, cgo


def visualize_pdb(pdb_path):
    # Load the PDB file
    cmd.load(pdb_path, "molecule")

    # Select and show only proteins and ligands
    cmd.select("protein_or_ligand", "polymer.protein or organic")
    cmd.show("cartoon", "polymer.protein")
    cmd.show("sticks", "organic")
    cmd.hide("everything", "resn POP or hydrogens or inorganic")
    cmd.set_color("mdp_gray", [0.745, 0.745, 0.745])
    cmd.bg_color("white")
    cmd.select("polymer_protein", "polymer and protein")
    cmd.color("mdp_gray", "polymer_protein")


def visualize_clusters(json_path):
    count = 0
    with open(json_path, "r") as json_file:
        clusters = json.load(json_file)
    for prop in clusters:
        pos0 = [prop["coord1"][0], prop["coord1"][1], prop["coord1"][2]]
        pos1 = [prop["coord2"][0], prop["coord2"][1], prop["coord2"][2]]
        r = prop["radius"]
        color = [prop["color"][0], prop["color"][1], prop["color"][2]]
        cylinder = [cgo.CYLINDER, *pos0, *pos1, r, *color, *color]
        cmd.load_cgo(cylinder, f"path_{count}")
        count += 1


# The function to be called by PyMOL with provided arguments
def main(pdb_path, json_path):
    print(f"pdb_path: {pdb_path}")
    print(f"json_path: {json_path}")
    visualize_pdb(pdb_path)
    visualize_clusters(json_path)
    cmd.center("polymer_protein")


cmd.extend("mdpath", main)
