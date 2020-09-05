import pymesh   # order somehow matters
from subprocess import Popen, PIPE
import os
import MDAnalysis
import numpy as np
import glob
import argparse
from Bio.PDB import *
from xyzrn import output_pdb_as_xyzrn
from read_msms import read_msms
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import *
from fixmesh import fix_mesh
from masif_opts import masif_opts
from compute_normal import compute_normal
from save_ply import save_ply
from read_data_from_surface import read_data_from_surface


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)

def get_masif(fbase):
    try:
        fbase = fbase.split(".")[0]
        print(fbase)
        args = ["reduce", "-Trim", fbase+".pdb"]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        outfile = open(fbase+"_ignh.pdb", "w")
        outfile.write(stdout.decode('utf-8').rstrip())
        outfile.close()
        # Now add them again.
        args = ["reduce", "-HIS", fbase+"_ignh.pdb"]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        outfile = open(fbase+".pdb", "w")
        outfile.write(stdout.decode('utf-8'))
        outfile.close()

        # os.system("sed -i \'/USER/d\' " + fbase+".pdb")

        output_pdb_as_xyzrn(fbase+".pdb", fbase+".xyzrn")

        FNULL = open(os.devnull, 'w')
        args = [os.environ['MSMS_BIN'], "-density", "3.0", "-hdensity", "3.0", "-probe",\
                        "1.5", "-if", fbase+".xyzrn","-of", fbase+"", "-af", fbase+""]

        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

        vertices, faces, normals, names = read_msms(fbase+"")
        areas = {}
        ses_file = open(fbase+""+".area")
        next(ses_file) # ignore header line
        for line in ses_file:
            fields = line.split()
            areas[fields[3]] = fields[1]

        vertex_hbond = computeCharges(fbase+"", vertices, names)

        vertex_hphobicity = computeHydrophobicity(fbase, names)

        mesh = pymesh.form_mesh(vertices, faces)
        regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices,\
                vertex_hbond, masif_opts)

        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices,\
                vertex_hphobicity, masif_opts)

        old_resids = np.load(fbase+'.npy')
        print("Old resid length: ", len(old_resids))
        new_resid = assignChargesToNewMesh(regular_mesh.vertices, vertices,\
                old_resids, None)
        print("Length of new resids: ", len(new_resid))
        np.save(fbase+'.npy', new_resid)

        pdb2pqr_path = "/home/lutian5/scratch/work/pdb2pqr/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
        args = [pdb2pqr_path,"--ff=amber","--whitespace","--noopt","--apbs-input",fbase+".pdb",fbase+""]   # changed ff to amber from parse for nucleic acids handling
        # args = [os.environ['PDB2PQR_BIN'],"--ff=amber","--whitespace","--noopt","--apbs-input",fbase+".pdb",fbase+""]   # changed ff to amber from parse for nucleic acids handling
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

        args = [os.environ['APBS_BIN'], fbase+"" + ".in"]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

        vertfile = open(fbase+".csv", "w")
        for vert in regular_mesh.vertices:
            vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))

        vertfile.close()

        args = [os.environ['MULTIVALUE_BIN'], fbase+"" + ".csv", fbase+"" + ".dx", fbase+"" + "_out.csv"]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

        # Read the charge file
        chargefile = open(fbase+"" + "_out.csv")
        charges = np.array([0.0] * len(regular_mesh.vertices))
        for ix, line in enumerate(chargefile.readlines()):
            charges[ix] = float(line.split(",")[3])

        save_ply(fbase+".ply", regular_mesh.vertices,\
                  regular_mesh.faces, normals=vertex_normal, charges=charges,\
                  normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)


        ### 04-masif_precompute.py
        params = masif_opts["ppi_search"]
        input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(fbase+".ply", params)

        np.save(fbase+'_rho_wrt_center', rho)
        np.save(fbase+'_theta_wrt_center', theta)
        np.save(fbase+'_input_feat', input_feat)
        np.save(fbase+'_mask', mask)
        np.save(fbase+'_list_indices', neigh_indices)
        np.save(fbase+'_iface_labels', iface_labels)
        # Save x, y, z
        np.save(fbase+'_X.npy', verts[:,0])
        np.save(fbase+'_Y.npy', verts[:,1])
        np.save(fbase+'_Z.npy', verts[:,2])


        ### masif_ppi_search_comp_desc.py

        print("Reading pre-trained network")
        from MaSIF_ppi_search import MaSIF_ppi_search

        learning_obj = MaSIF_ppi_search(
            params["max_distance"],
            n_thetas=16,
            n_rhos=5,
            n_rotations=16,
            idx_gpu="/gpu:0",
            feat_mask=params["feat_mask"],
        )
        learning_obj.saver.restore(learning_obj.session, params["model_dir"] + "model")

        from train_ppi_search import compute_val_test_desc

        p1_rho_wrt_center = np.load(fbase+"_rho_wrt_center.npy")
        p1_theta_wrt_center = np.load(fbase+"_theta_wrt_center.npy")
        p1_input_feat = np.load(fbase+"_input_feat.npy")
        p1_input_feat = mask_input_feat(p1_input_feat, params["feat_mask"])
        p1_mask = np.load(fbase+"_mask.npy")
        idx1 = np.array(range(len(p1_rho_wrt_center)))
        desc1_str = compute_val_test_desc(
            learning_obj,
            idx1,
            p1_rho_wrt_center,
            p1_theta_wrt_center,
            p1_input_feat,
            p1_mask,
            batch_size=1000,
            flip=False,
        )

        np.save(fbase+"_desc_straight.npy", desc1_str)
        return 0
    except:
        return 1

def get_desc(line, row, fields):
    receptor_name = "pdb{}_{}.pdb".format(fields[0], fields[1])
    receptor_base = receptor_name.split(".")[0]
    peptide_name = "pdb{}_{}.pdb".format(fields[0], fields[2])
    peptide_base = peptide_name.split(".")[0]
    u = MDAnalysis.Universe(peptide_name)
    atoms = u.select_atoms('protein and backbone and name CA and not altloc B and not altloc C').positions
    start = int(fields[3]) - 1
    length = int(fields[-2])
    end = start + length + 1
    atoms = atoms[start:end]
    center_coords = atoms[int(length/2)]
    # print(center_coords)
    # get idx of closest vertex to center_coords
    # X = np.load("{}_X.npy".format(receptor_base), allow_pickle=True).reshape(-1,1)
    # Y = np.load("{}_Y.npy".format(receptor_base), allow_pickle=True).reshape(-1,1)
    # Z = np.load("{}_Z.npy".format(receptor_base), allow_pickle=True).reshape(-1,1)
    # vertex_coords = np.concatenate((X,Y,Z), axis=1)
    vertex_coords = np.genfromtxt('{}.csv'.format(receptor_base), delimiter=',')
    # print(vertex_coords.shape)
    diff = np.linalg.norm(vertex_coords - center_coords, axis=1)
    idx = np.argsort(diff)[:5]
    # print("vertex index: ", idx)
    desc = np.load("{}_desc_straight.npy".format(receptor_base), allow_pickle=True)[idx]
    np.save("{}.npy".format(row), desc)

def process_data(table_fname):
    with open(table_fname, 'r') as f:
        suffix = table_fname.split(".")[0][5:]
        line = f.readline()
        # line = f.readline()  # skip first line
        row = 0
        curr_id = ""
        while line:
            fields = line.split(",")
            receptor_name = "pdb{}_{}.pdb".format(fields[0], fields[1])
            peptide_name = "pdb{}_{}.pdb".format(fields[0], fields[2])
            folder_name = fields[0][1:3]
            os.system("cp ../{}/{} .".format(folder_name, receptor_name))
            os.system("cp ../{}/{} .".format(folder_name, peptide_name))
            os.system("cp ../{}/filtered/{} .".format(folder_name, receptor_name))
            os.system("cp ../{}/filtered/{} .".format(folder_name, peptide_name))
            if os.path.exists(receptor_name):
                save_name = "{}-row-{}".format(row, suffix)
                if fields[0] != curr_id:
                    if row != 0:
                        os.system("rm pdb{}*".format(curr_id))
                    curr_id = fields[0]
                    get_masif(receptor_name)
                get_desc(line, save_name, fields)
            row += 1
            line = f.readline()

if __name__ == '__main__':
    pass
    ## parse command line
    parser = argparse.ArgumentParser(description='Enter table file name')
    # required params
    parser.add_argument('-Sfname', '--tablename', nargs=1, type=str, const=None,
                            help="Name of table*.csv file",
                            dest='table_fname')

    args = parser.parse_args()
    table_fname = args.table_fname[0]
    process_data(table_fname)