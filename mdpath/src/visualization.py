"""Visualization --- :mod:`mdpath.scr.visualization`
==============================================================================

This module contains the class `MDPathVisualize` which contains all visualization functions for the MDPath package.
The class only contains static methods for visualization purposes.

Classes
--------

:class:`MDPathVisualize`
"""

from Bio import PDB
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import MDAnalysis as mda
import requests
import json
from collections import defaultdict
from scipy.interpolate import CubicSpline
from stl import mesh

Colors = [
    [0.1216, 0.4667, 0.7059],
    [0.1725, 0.6647, 0.1725],
    [0.8392, 0.1529, 0.1569],
    [0.5804, 0.4039, 0.7412],
    [0.5490, 0.3373, 0.2941],
    [0.8902, 0.4667, 0.7608],
    [1.0000, 0.4980, 0.0549],
]

AAMAPPING = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class MDPathVisualize:
    """Methods for visualization within the MDPath package.

    Attributes:
        None (only static methods)"""

    def __init__(self) -> None:
        pass

    @staticmethod
    def residue_CA_coordinates(pdb_file: str, end: int) -> dict:
        """Collects CA atom coordinates for residues.

        Args:
            pdb_file (str): Path to PDB file.

            end (int): Last residue to consider.

        Returns:
            residue_coordinates_dict (dict): Dictionary with residue number as key and CA atom coordinates as value.
        """
        residue_coordinates_dict = {}
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", pdb_file)
        residues = [
            res for res in structure.get_residues() if PDB.Polypeptide.is_aa(res)
        ]
        for res in tqdm(residues, desc="\033[1mProcessing residues: \033[0m"):
            res_id = res.get_id()[1]
            if res_id <= end:
                for atom in res:
                    if atom.name == "CA":
                        if res_id not in residue_coordinates_dict:
                            residue_coordinates_dict[res_id] = []
                        residue_coordinates_dict[res_id].append(atom.coord)
        return residue_coordinates_dict

    @staticmethod
    def cluster_prep_for_visualisation(cluster: list, pdb_file: str) -> list:
        """Prepares pathway clusters for visualisation.

        Args:
            cluster (list): Cluster of pathways.

            pdb_file (str): Path to PDB file.

        Returns:
            cluster (list): Cluster of pathways with CA atom coordinates.
        """
        new_cluster = []
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", pdb_file)

        for pathway in cluster:
            pathways = []
            for residue in pathway:
                res_id = ("", residue, "")
                try:
                    res = structure[0][res_id]
                    atom = res["CA"]
                    coord = tuple(atom.get_coord())
                    pathways.append(coord)
                except KeyError:
                    print(f"Residue {res_id} not found.")
            new_cluster.append(pathways)

        return new_cluster

    @staticmethod
    def apply_backtracking(original_dict: dict, translation_dict: dict) -> dict:
        """Backtracks the original dictionary with a translation dictionary.

        Args:
            original_dict (dict): Cluster pathways dictionary.

            translation_dict (dict): Residue coordinates dictionary.

        Returns:
            updated_dict (dict): Updated cluster pathways dictionary with residue coordinates.
        """
        updated_dict = original_dict.copy()

        for key, lists_of_lists in original_dict.items():
            for i, inner_list in enumerate(lists_of_lists):
                for j, item in enumerate(inner_list):
                    if item in translation_dict:
                        updated_dict[key][i][j] = translation_dict[item]

        return updated_dict

    @staticmethod
    def format_dict(updated_dict: dict) -> dict:
        """Reformats the dictionary to be JSON serializable.

        Args:
            updated_dict (dict): Dictionary to be reformatted.

        Returns:
            transformed_dict (dict): Reformatted dictionary.
        """

        def transform_list(nested_list):
            transformed = []
            for item in nested_list:
                if isinstance(item, np.ndarray):
                    transformed.append(item.tolist())
                elif isinstance(item, list):
                    transformed.append(transform_list(item))  # Append instead of extend
                else:
                    transformed.append(item)
            return transformed

        transformed_dict = {
            key: transform_list(value) for key, value in updated_dict.items()
        }
        return transformed_dict

    @staticmethod
    def visualise_graph(graph: nx.Graph, k=0.1, node_size=200) -> None:
        """Draws residue graph to PNG file.

        Args:
            graph (nx.Graph): Residue graph.

            k (float, optional): Distance between individual nodes. Defaults to 0.1.

            node_size (int, optional): Size of individual nodes. Defaults to 200.
        """
        labels = {i: str(i) for i in graph.nodes()}
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(graph, k=k)
        nx.draw(
            graph,
            pos,
            node_size=node_size,
            with_labels=True,
            labels=labels,
            font_size=8,
            edge_color="gray",
            node_color="skyblue",
        )
        plt.savefig("graph.png", dpi=300, bbox_inches="tight")

    @staticmethod
    def precompute_path_properties(json_data: dict) -> list:
        """Precomputes path properties for quicker visualization in Jupyter notebook.

        Args:
            json_data (dict): Cluster data with pathways and CA atom coordinates.

        Returns:
            path_properties (list): List of path properties. Contains clusterid, pathway index, path segment index, coordinates, color, radius, and path number.
        """
        cluster_colors = {}
        color_index = 0
        path_properties = []

        for clusterid, cluster in json_data.items():
            cluster_colors[clusterid] = Colors[color_index % len(Colors)]
            color_index += 1
            coord_pair_counts = {}
            path_number = 1

            for pathway_index, pathway in enumerate(cluster):
                for i in range(len(pathway) - 1):
                    coord1 = pathway[i][0]
                    coord2 = pathway[i + 1][0]
                    if (
                        isinstance(coord1, list)
                        and isinstance(coord2, list)
                        and len(coord1) == 3
                        and len(coord2) == 3
                    ):
                        coord_pair = (tuple(coord1), tuple(coord2))
                        if coord_pair not in coord_pair_counts:
                            coord_pair_counts[coord_pair] = 0
                        coord_pair_counts[coord_pair] += 1
                        radius = 0.015 + 0.015 * (coord_pair_counts[coord_pair] - 1)
                        color = cluster_colors[clusterid]

                        path_properties.append(
                            {
                                "clusterid": clusterid,
                                "pathway_index": pathway_index,
                                "path_segment_index": i,
                                "coord1": coord1,
                                "coord2": coord2,
                                "color": color,
                                "radius": radius,
                                "path_number": path_number,
                            }
                        )

                        path_number += 1
                    else:
                        print(
                            f"Ignoring pathway {pathway} as it does not fulfill the coordinate format."
                        )
        return path_properties

    @staticmethod
    def precompute_cluster_properties_quick(json_data: dict) -> list:
        """Precomputes cluster properties for quicker visualization in Jupyter notebook.

        Args:
            json_data (dict): Cluster data with pathways and CA atom coordinates.

        Returns:
            cluster_properties (list): List of cluster properties. Contains clusterid,coordinates, color, and radius.
        """
        cluster_colors = {}
        color_index = 0
        cluster_properties = []

        for clusterid, cluster in json_data.items():
            cluster_colors[clusterid] = Colors[color_index % len(Colors)]
            color_index += 1
            coord_pair_counts = {}

            for pathway_index, pathway in enumerate(cluster):
                for i in range(len(pathway) - 1):
                    coord1 = pathway[i][0]
                    coord2 = pathway[i + 1][0]
                    if (
                        isinstance(coord1, list)
                        and isinstance(coord2, list)
                        and len(coord1) == 3
                        and len(coord2) == 3
                    ):
                        coord_pair = (tuple(coord1), tuple(coord2))
                        if coord_pair not in coord_pair_counts:
                            coord_pair_counts[coord_pair] = 0
                        coord_pair_counts[coord_pair] += 1
                        radius = 0.015 + 0.015 * (coord_pair_counts[coord_pair] - 1)
                        color = cluster_colors[clusterid]

                        cluster_properties.append(
                            {
                                "clusterid": clusterid,
                                "coord1": coord1,
                                "coord2": coord2,
                                "color": color,
                                "radius": radius,
                            }
                        )
                    else:
                        print(
                            f"Ignoring pathway {pathway} as it does not fulfill the coordinate format."
                        )
        return cluster_properties

    @staticmethod
    def remove_non_protein(input_pdb: str, output_pdb: str) -> None:
        """Function to remove non-protein atoms (e.g., water, ligands, ions) from a PDB file
        and write only the protein atoms to a new PDB file.

        Args:
            input_pdb (str): Path to the input PDB file.

            output_pdb (str): Path to the output PDB file to save the protein-only structure.
        """
        sys = mda.Universe(input_pdb)
        protein = sys.select_atoms("protein")
        protein.write(output_pdb)

    @staticmethod
    def assign_generic_numbers(
        pdb_file_path: str, output_file_path: str = "numbered_structure.pdb"
    ) -> None:
        """Assigns generic numbers to residues in a PDB file by querying the gpcrdb.org.

        Args:
            pdb_file_path (str): Path to the PDB file.

            output_file_path (str, optional): Path to save the new PDB file with generic numbers. Defaults to "numbered_structure.pdb".
        """
        url = "https://gpcrdb.org/services/structure/assign_generic_numbers"
        with open(pdb_file_path, "rb") as pdb_file:
            files = {"pdb_file": pdb_file}
            response = requests.post(url, files=files)
        if response.status_code == 200:
            with open(output_file_path, "w") as output_file:
                output_file.write(response.text)
            print(f"New PDB file saved as {output_file_path}")
        else:
            print(f"Failed to process the file: {response.status_code}")

    @staticmethod
    def parse_pdb_and_create_dictionary(pdb_file_path: str) -> dict:
        """Parses a PDB file and creates a dictionary with residue numbers, generic numbers, and amino acids.

        Args:
            pdb_file_path (str): Path to the PDB file.

        Returns:
            residue_dict (dict): Dictionary with residue numbers, generic numbers, and amino acids.
        """
        processed_residues = []
        residue_dict = {}
        last_generic_number = 1
        with open(pdb_file_path, "r") as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM"):
                    residue_number = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                    amino_acid = line[17:20].strip()
                    if residue_number not in processed_residues:
                        if b_factor > 0.1 and b_factor < 8.99:
                            generic_number = str(f"{b_factor:.2f}").replace(".", "x")
                            match = re.match(r"(\d)[x\.](\d+)", generic_number)
                            if int(match.group(1)) in range(1, 8):
                                last_generic_number = int(match.group(1))
                            processed_residues.append(residue_number)
                        else:
                            if last_generic_number == 7:
                                generic_number = (
                                    f"{last_generic_number+1}x{residue_number}"
                                )
                            else:
                                generic_number = f"{last_generic_number}{last_generic_number+1}x{residue_number}"
                        if amino_acid in AAMAPPING:
                            residue_dict[residue_number] = {
                                "generic_number": generic_number,
                                "amino_acid": AAMAPPING[amino_acid],
                            }
        return residue_dict

    @staticmethod
    def assign_generic_numbers_paths(
        cluster_pathways: dict, generic_number_dict: dict
    ) -> tuple:
        """Assigns generic numbers to residues in the cluster pathways.

        Args:
            cluster_pathways (dict): Dictionary with cluster pathways.

            generic_number_dict (dict): Dictionary with residue numbers, generic numbers, and amino acids.

        Returns:
            updated_cluster_residues (dict): Updated dictionary with cluster pathways and generic numbers.

            no_genetic_number_list (list): List of residue numbers with no generic numbers.
        """
        updated_cluster_residues = {}
        no_genetic_number_list = []
        for cluster_id, residue_lists in cluster_pathways.items():
            updated_residue_lists = []
            for residue_list in residue_lists:
                updated_residue_list = []
                for residue_number in residue_list:
                    try:
                        updated_residue_list.append(
                            generic_number_dict[residue_number]["generic_number"]
                        )
                    except KeyError:
                        no_genetic_number_list.append(residue_number)
                updated_residue_lists.append(updated_residue_list)
            updated_cluster_residues[cluster_id] = updated_residue_lists
        no_genetic_number_list = set(no_genetic_number_list)
        return updated_cluster_residues, no_genetic_number_list

    @staticmethod
    def draw_column(
        draw: ImageDraw.Draw,
        col: int,
        res: list,
        label: str,
        circle_positions: dict,
        circle_diameter: int,
        padding: int,
        column_width: int,
        height: int,
        font: ImageFont,
        title_font: ImageFont,
        align: str = "top",
    ) -> None:
        """Draws a column in the given pillow drawing context corresponding to a TM region or loop region with a label, a rectangle,
        and circles with genetic numbers corresponding to residues in this region that are part of a path.

        Args:
            draw (ImageDraw.Draw): The drawing context.
            col (int): The column index (1-based).
            res (list): A list of tuples containing data to be visualized, where each tuple contains an identifier and a genetic number.
            label (str): The label for the column.
            circle_positions (dict): A dictionary to store the positions of the circles, keyed by genetic number.
            circle_diameter (int): The diameter of the circles to be drawn.
            padding (int): The padding between elements.
            column_width (int): The width of the column.
            height (int): The height of the drawing area.
            font (ImageFont): The font to be used for the genetic numbers.
            title_font (ImageFont): The font to be used for the column label.
            align (str, optional): The alignment of the circles within the column. Can be 'top' or 'bottom'. Defaults to 'top'.
        """

        x = (col - 1) * (column_width + padding) + padding

        draw.text(
            ((x + column_width // 2) - 12, 10),
            f"{label}",
            fill="black",
            font=title_font,
        )

        draw.rectangle([x, 40, x + column_width, height - padding], outline="black")

        # Draw circles and labels
        for i, (_, genetic_number) in enumerate(res):
            if align == "top":
                circle_y = 80 + i * (circle_diameter + padding)
            elif align == "bottom":
                circle_y = (
                    height - padding - i * (circle_diameter + padding) - circle_diameter
                )
            else:
                raise ValueError(
                    "Invalid value for align. It should be 'top' or 'bottom'."
                )
            circle_x = x + column_width // 2
            draw.ellipse(
                [
                    circle_x - circle_diameter // 2,
                    circle_y - circle_diameter // 2,
                    circle_x + circle_diameter // 2,
                    circle_y + circle_diameter // 2,
                ],
                outline="black",
            )

            # Draw genetic number
            draw.text(
                (circle_x - 28, circle_y - 8),
                f"{genetic_number}",
                fill="black",
                font=font,
            )
            circle_positions[genetic_number] = (circle_x, circle_y)

    @staticmethod
    def create_gpcr_2d_path_vis(
        updated_cluster_residues: dict,
        cutoff_percentage: int = 0,
        image_name: str = "GPCR_2D_pathways",
        fontsize_tm: int = 20,
        fontsize_numbers: int = 18,
        fontfile: str = None,
    ) -> None:
        """Creates a 2D visualization of pathways within a GPCR based on the provided cluster residues.

        Args:
            updated_cluster_residues (dict): A dictionary where keys are cluster identifiers and values are lists of paths. Each path is a list of residue identifiers in the format 'TMx.y'.

            cutoff_percentage (int, optional): The percentage cutoff for drawing connections between residues. Only connections with a frequency above this percentage will be drawn. Defaults to 0.

            image_name (str, optional): The base name for the output image files. Defaults to "GPCR_2D_pathways".

            fontsize_tm (int, optional): The font size for the transmembrane (TM) labels. Defaults to 20.

            fontsize_numbers (int, optional): The font size for the residue numbers. Defaults to 18.

            fontfile (str, optional): The path to a font file to use for text rendering. If None, the default Pillow font is used. Defaults to None.

        Returns:
            None. The function saves the generated images to disk with filenames based on the provided image_name and cluster identifiers.
        """

        for cluster in updated_cluster_residues.keys():

            # Data preparation
            tm_data = {i: [] for i in range(1, 8)}
            icl_data = []
            ecl_data = []
            for path in updated_cluster_residues[cluster]:
                for res in path:
                    match = re.match(r"(\d+)x(\d+)", res)
                    if match:
                        tm_number = int(match.group(1))
                        position = int(match.group(2))
                        if 1 <= tm_number <= 7:
                            tm_data[tm_number].append((position, res))
                        elif tm_number in [12, 34, 56, 8]:
                            icl_data.append((position, res))
                        elif tm_number in [23, 45, 67]:
                            ecl_data.append((position, res))

            # Remove duplicate values and sort
            for tm_number, values in tm_data.items():
                tm_data[tm_number] = list(set(values))
            icl_data = list(set(icl_data))
            ecl_data = list(set(ecl_data))
            for tm in tm_data.values():
                tm.sort(key=lambda x: x[0])
            icl_data.sort(key=lambda x: x[0])
            ecl_data.sort(key=lambda x: x[0])

            max_circles = max(
                max(len(res) for res in tm_data.values()), len(icl_data), len(ecl_data)
            )

            # Image size
            circle_diameter = 75
            padding = 40
            column_width = 100
            width = (len(tm_data) + 2) * (column_width + padding) + padding
            height = max_circles * (circle_diameter + padding) + padding * 2

            image = Image.new("RGB", (width, height), color="white")
            draw = ImageDraw.Draw(image)

            # Load a font
            if fontfile:
                try:
                    font = ImageFont.truetype(fontfile, fontsize_numbers)
                    title_font = ImageFont.truetype(fontfile, fontsize_tm)
                except IOError:
                    print(
                        f"Could not load font file {fontfile}. Using pillow default font."
                    )
                    font = ImageFont.load_default(size=fontsize_numbers)
                    title_font = ImageFont.load_default(size=fontsize_tm)
            else:
                print(f"No font file provided. Using pillow default font.")
                font = ImageFont.load_default(size=fontsize_numbers)
                title_font = ImageFont.load_default(size=fontsize_tm)

            circle_positions = {}
            for col, (tm_number, res) in enumerate(tm_data.items(), start=1):
                MDPathVisualize.draw_column(
                    draw,
                    col,
                    res,
                    f"TM{tm_number}",
                    circle_positions,
                    circle_diameter,
                    padding,
                    column_width,
                    height,
                    font,
                    title_font,
                )
            MDPathVisualize.draw_column(
                draw,
                len(tm_data) + 1,
                icl_data,
                "IC",
                circle_positions,
                circle_diameter,
                padding,
                column_width,
                height,
                font,
                title_font,
                align="bottom",
            )
            MDPathVisualize.draw_column(
                draw,
                len(tm_data) + 2,
                ecl_data,
                "EC",
                circle_positions,
                circle_diameter,
                padding,
                column_width,
                height,
                font,
                title_font,
            )

            # Count the frequency of each path and calculate cutoff
            connection_counts = Counter()
            for path in updated_cluster_residues[cluster]:
                for i in range(len(path) - 1):
                    connection = tuple(sorted([path[i], path[i + 1]]))
                    connection_counts[connection] += 1
            max_count = max(connection_counts.values()) if connection_counts else 1
            cutoff_count = (cutoff_percentage / 100) * max_count

            # Draw lines between connected residues with varying thickness
            for path in updated_cluster_residues[cluster]:
                for i in range(len(path) - 1):
                    current_res = path[i]
                    next_res = path[i + 1]

                    if current_res in circle_positions and next_res in circle_positions:
                        start = circle_positions[current_res]
                        end = circle_positions[next_res]

                        # Get the count for this connection
                        connection = tuple(sorted([current_res, next_res]))
                        count = connection_counts[connection]

                        # Only draw the line if the count is above the cutoff
                        if count >= cutoff_count:
                            thickness = max(1, min(5, int((count / max_count) * 10)))
                            draw.line([start, end], fill="blue", width=thickness)
            # Save the image
            image.save(f"{image_name}_cluster_{cluster}.png")
            print(f"Image saved as {image_name}_cluster_{cluster}.png")

    @staticmethod
    def create_splines(
        json_path: str,
    ) -> None:

        def group_clusters(data):
            """Groups JSON elements by their clusterid."""
            clusters = defaultdict(list)
            for item in data:
                clusters[item["clusterid"]].append(item)
            return clusters

        def find_connected_paths(clusters):
            """Find continuous paths by connecting matching coordinates and their radii."""
            paths = []
            used_indices = set()
            for i, cluster in enumerate(clusters):
                if i in used_indices:
                    continue

                current_path = [cluster["coord1"], cluster["coord2"]]
                current_radii = [cluster["radius"], cluster["radius"]]
                current_color = cluster["color"]
                used_indices.add(i)

                last_coord = cluster["coord2"]
                while True:
                    found_next = False
                    for j, next_cluster in enumerate(clusters):
                        if j in used_indices:
                            continue

                        if np.allclose(next_cluster["coord1"], last_coord):
                            current_path.append(next_cluster["coord2"])
                            current_radii.append(next_cluster["radius"])
                            last_coord = next_cluster["coord2"]
                            used_indices.add(j)
                            found_next = True
                            break

                    if not found_next:
                        break

                paths.append(
                    {
                        "coords": current_path,
                        "radii": current_radii,
                        "color": current_color,
                    }
                )

            return paths

        def create_spline(coord_list, radii_list):
            """Create splines for both coordinates and radii."""
            coords = np.array(coord_list)
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            t = np.linspace(0, 1, len(coords))

            spline_x = CubicSpline(t, x)
            spline_y = CubicSpline(t, y)
            spline_z = CubicSpline(t, z)

            spline_r = CubicSpline(t, radii_list)
            t_fine = np.linspace(0, 1, 100)
            x_fine = spline_x(t_fine)
            y_fine = spline_y(t_fine)
            z_fine = spline_z(t_fine)
            r_fine = spline_r(t_fine)

            points = np.vstack((x_fine, y_fine, z_fine)).T
            return points, r_fine

        def generate_path_faces(path):
            """Generate faces with varying radii along the path."""
            faces = []

            spline_points, radii = create_spline(path["coords"], path["radii"])

            segments = len(spline_points) - 1
            for i in range(segments):
                for j in range(8):
                    theta1 = 2 * np.pi * j / 8
                    theta2 = 2 * np.pi * (j + 1) / 8

                    p1 = spline_points[i] + radii[i] * np.array(
                        [np.cos(theta1), np.sin(theta1), 0]
                    )
                    p2 = spline_points[i + 1] + radii[i + 1] * np.array(
                        [np.cos(theta1), np.sin(theta1), 0]
                    )
                    p3 = spline_points[i] + radii[i] * np.array(
                        [np.cos(theta2), np.sin(theta2), 0]
                    )
                    p4 = spline_points[i + 1] + radii[i + 1] * np.array(
                        [np.cos(theta2), np.sin(theta2), 0]
                    )

                    faces.append([p1, p2, p3])
                    faces.append([p2, p4, p3])

            return faces

        def process_cluster(cluster_data, output_file):
            """Process a single cluster and save it as an STL file."""
            paths = find_connected_paths(cluster_data)
            all_faces = []
            for path in paths:
                path_faces = generate_path_faces(path)
                all_faces.extend(path_faces)

            num_faces = len(all_faces)
            data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
            for i, face in enumerate(all_faces):
                data[i]["vectors"] = np.array(face)

            combined_mesh = mesh.Mesh(data)
            combined_mesh.save(output_file)

        directory = os.path.dirname(json_path)
        mesh_folder = os.path.join(directory, "cluster_meshes")
        os.makedirs(mesh_folder, exist_ok=True)

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        clusters_dict = group_clusters(data)
        for cluster_id, cluster_data in clusters_dict.items():
            cluster_json_path = os.path.join(mesh_folder, f"cluster_{cluster_id}.json")
            with open(cluster_json_path, "w") as f:
                json.dump(cluster_data, f, indent=4)

            cluster_stl_path = os.path.join(mesh_folder, f"cluster_{cluster_id}.stl")
            process_cluster(cluster_data, cluster_stl_path)
            print(f"Processed cluster {cluster_id}: JSON and STL files saved")
