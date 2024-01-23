import numpy as np
from scipy.spatial import *
from tqdm import tqdm


def planes(vc): #[ncells,tri,ndim]
    u = vc[:,1,:]-vc[:,0,:]
    v = vc[:,2,:]-vc[:,0,:] #(ncells, ndim)
    normal=np.cross(u,v)
    r0=vc[:,0,:]
    normal=normal/np.sqrt(np.sum(normal**2,axis=-1))[:,None] # normalize vectors
    return r0, normal

def projections(vc, verts):
    r0, normal=planes(vc)
    dists=np.sum((verts[:,None,:]-r0[None,:,:])*normal[None,:,:], axis=-1)
    projs = verts[:,None,:] - dists[:,:,None]*normal[None,:,:]
        
    return dists, projs

def project(verts,a,b):
    normal=b-a
    dists=np.sum((verts[:,None,:]-a[None,:,:])*normal[None,:,:], axis=-1)
    projs = a[None,:,:] + dists[:,:,None]*normal[None,:,:]
    return projs

def bis(p1,p2,p3):
    a = np.sum((p1-p2)**2,axis=-1,keepdims=True)
    b = np.sum((p2-p3)**2,axis=-1,keepdims=True)
    c = np.sum((p1-p3)**2,axis=-1,keepdims=True)
    bis=(a*p3+b*p1+c*p2)/(a+b+c)
    return (bis)
    
def get_angle(p1,p2,p3):
    a = p2-p3
    c = p2-p1
    angle_b = (np.sum(a*c, axis=-1) / np.sqrt(np.sum(a*a, axis=-1)*np.sum(c*c, axis=-1))).clip(-1,1)
    return np.arccos(angle_b)

def max_angle(vc,verts):
    r0, normal=planes(vc)
    dists=np.sum((verts[:,None,:]-r0[None,:,:])*normal[None,:,:], axis=-1)
    projs = verts[:,None,:] - dists[:,:,None]*normal[None,:,:]
    
    proj1=project(verts,vc[:,1,:], vc[:,2,:])
    proj2=project(verts,vc[:,0,:], vc[:,2,:])
    proj3=project(verts,vc[:,0,:], vc[:,1,:])

    a1=get_angle(projs,proj1,verts[:,None,:])
    a1=np.where(np.sum((proj1-projs)*(vc[None,:,0,:]-projs), axis=-1)<0,a1,np.pi-a1)
    a2=get_angle(projs,proj2,verts[:,None,:])
    a2=np.where(np.sum((proj2-projs)*(vc[None,:,1,:]-projs), axis=-1)<0,a1,np.pi-a1)
    a3=get_angle(projs,proj3,verts[:,None,:])
    a3=np.where(np.sum((proj3-projs)*(vc[None,:,2,:]-projs), axis=-1)<0,a1,np.pi-a1)
    angle=np.max(np.array([a1,a2,a3]),axis=0)
    return angle

def split_triangles(verts):
    cells=ConvexHull(verts).simplices
    print('Verts left:',verts.shape[0])
    print('Num cells:',cells.shape[0])
    mask=np.delete(np.arange(verts.shape[0]),np.unique(cells))
    print('Verts left:',mask.shape[0])
    for i in range(1):
        d=max_angle(verts[cells],verts[mask])
        #d, _=projections(verts[cells],verts[mask])
        nearest_dots=np.argmin(d,axis=0)
        mins=np.min(d,axis=0)
        nearest_cells=np.argsort(mins)
        nearest_cells=nearest_cells[:(mins<(np.pi/4)).sum()]
        nearest_dots=nearest_dots[nearest_cells]
        nearest_dots, idx=np.unique(nearest_dots, return_index=True)#?
        nearest_dots=mask[nearest_dots]
        nearest_cells=nearest_cells[idx]
        c1=cells[nearest_cells]
        c2=np.array([[c1[:,0],nearest_dots,c1[:,1]],
                       [c1[:,0],nearest_dots,c1[:,2]],
                       [c1[:,1],nearest_dots,c1[:,2]]]) 
        c2=c2.transpose(0,2,1).reshape((-1,3))
        cells=np.delete(cells,nearest_cells, axis=0)
        cells=np.concatenate((cells,c2))
        mask=np.delete(np.arange(verts.shape[0]),np.unique(cells))
        print('Num cells:',cells.shape[0])
        print('Verts left:',mask.shape[0])
        if mask.shape[0]<1:
            break
    return cells

def find_connected_components(nei):
    nodes=np.arange(nei.shape[0])
    list_components=np.zeros(nei.shape[0], dtype=int)
    i=0
    while np.sum(list_components==0)>0:
        i+=1
        list_components[np.argmin(list_components)]=i
        k=1
        while True:
            tocol=nei[list_components==i,:].reshape(-1)
            list_components[tocol]=i
            if np.sum(list_components==i)==k:
                break
            else:
                k=np.sum(list_components==i)
    return list_components

def remove_non_connected(tetr,nei):
    for v in range(tetr.max()):
        t=np.arange(tetr.shape[0])[(tetr==v).any(axis=1)]
        


def get_edges(v1, v2, v3, v4):
    e1=np.sqrt(np.sum((v2-v1)**2, axis=-1))
    e2=np.sqrt(np.sum((v3-v1)**2, axis=-1))
    e3=np.sqrt(np.sum((v4-v1)**2, axis=-1))
    e4=np.sqrt(np.sum((v3-v2)**2, axis=-1))
    e5=np.sqrt(np.sum((v4-v2)**2, axis=-1))
    e6=np.sqrt(np.sum((v4-v3)**2, axis=-1))
    y=np.array([[e4,e5,e6],
       [e2,e3,e6],
       [e1,e3,e5],
       [e1,e2,e4]])
    return np.transpose(y,(2,0,1))
    
def delaunay(verts, thr=4):
    dela=Delaunay(verts)
    
    tetr=dela.simplices
    nei=dela.neighbors
    edges=get_edges(verts[tetr[:,0]],verts[tetr[:,1]],verts[tetr[:,2]],verts[tetr[:,3]])
    order=np.flip(np.argsort(edges.sum(axis=(1,2))))
    print('Tetrahedrons:',tetr.shape[0])
    for i in tqdm(range(tetr.shape[0])):
        mask=order[(np.sum(nei[order]<0, axis=1)>0)&(np.sum(nei[order]<0, axis=1)<4)]
        c=0
        for t in mask:

            s1=edges[t,nei[t]==-1,:]
            s2=edges[t,nei[t]>-1,:]
            if s2.shape[0]==0 or s1.max()>=s2.max() or s1.max()>thr:
                nei[t,:]=[-1,-1,-1,-1]
                nei=np.where(nei==t,-1,nei)
                c+=1
        if c==0:
            break
        
    con=find_connected_components(nei)
    for i in range(1,np.max(con)+1):
        if np.sum(con==i)<10:
            nei[con==i,:]=[-1,-1,-1,-1]

    cells=np.stack((
        tetr[:,[1,2,3]],
        tetr[:,[0,2,3]],
        tetr[:,[0,1,3]],
        tetr[:,[0,1,2]],
        ), axis=1)
    
    mask=(nei<0)&(np.sum(nei<0, axis=1)<4)[:,None]
    
    skipped_verts=np.delete(np.arange(verts.shape[0]),np.unique(cells[mask,:]))
    
    cells=np.sort(cells[mask,:], axis=1)

    return(cells)

