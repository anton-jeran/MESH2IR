import pymeshlab as ml
import os

#6810
#13311

path ="Meshes"
move_path = "Simplified_Meshes"

os.mkdir(move_path)

# path ="/scratch/anton/template"
# move_path = "/scratch/anton/template_60000"

for subdir,dir,files in os.walk(path):
    for file in files:

        if(file.endswith(".obj")):
            
            f_path=os.path.join(subdir,file)
            m_path=os.path.join(move_path,file)
            TARGET_Faces=2000
         
        
            ms = ml.MeshSet()
            ms.load_new_mesh(f_path)
            m = ms.current_mesh()
            print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')


         
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=TARGET_Faces, preservenormal=True)
            print("Decimated to", TARGET_Faces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
           

            
            m = ms.current_mesh()
            print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
            ms.save_current_mesh(m_path)