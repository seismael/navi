import trimesh
import numpy as np
import sys
import importlib.util

def check_embree():
    print("=" * 60)
    print(f"Trimesh version: {trimesh.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Check if pyembree/embreex is installed in the environment
    embree_spec = importlib.util.find_spec("pyembree")
    embreex_spec = importlib.util.find_spec("embreex")
    
    if embree_spec:
        print(f"Found 'pyembree' at: {embree_spec.origin}")
    else:
        print("Package 'pyembree' NOT found.")

    if embreex_spec:
        print(f"Found 'embreex' at: {embreex_spec.origin}")
    else:
        print("Package 'embreex' NOT found.")

    # Check if embree is available via trimesh internal logic
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        print("Trimesh can import 'trimesh.ray.ray_pyembree'.")
    except ImportError as e:
        print(f"Trimesh CANNOT import 'trimesh.ray.ray_pyembree': {e}")

    print("-" * 60)

    # Create a simple mesh
    print("Creating test mesh (box)...")
    mesh = trimesh.creation.box()
    
    # Check what ray intersector is being used
    # When we access mesh.ray, trimesh initializes the intersector
    print("Initializing ray intersector (mesh.ray)...")
    ray_intersector = mesh.ray
    
    intersector_type = type(ray_intersector)
    intersector_module = intersector_type.__module__
    
    print(f"Ray intersector class: {intersector_type.__name__}")
    print(f"Ray intersector module: {intersector_module}")

    is_embree = "embree" in intersector_module.lower()

    if is_embree:
        print("SUCCESS: Embree is being used for ray tracing.")
    else:
        print("FAILURE: Embree is NOT being used. Fallback (likely rtree/numpy) is active.")

    print("-" * 60)
    print("Running functional test...")

    # Perform a simple raycast to ensure it works
    # Ray from [0,0,-2] pointing +Z [0,0,1] should hit the box at [0,0,-0.5]
    origins = np.array([[0, 0, -2]], dtype=np.float64)
    vectors = np.array([[0, 0, 1]], dtype=np.float64)
    
    try:
        # Measure time for a tiny check
        import time
        t0 = time.time()
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False
        )
        dt = time.time() - t0
        
        print(f"Raycast took {dt*1000:.3f} ms")
        print(f"Raycast hits: {len(locations)}")
        
        if len(locations) == 1:
             print("Raycast logic works (1 hit detected).")
             print(f"   Hit location: {locations[0]}")
        else:
             print(f"Raycast logic failed (expected 1 hit, got {len(locations)}).")
             
    except Exception as e:
        print(f"Raycast crashed: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)

if __name__ == "__main__":
    check_embree()
