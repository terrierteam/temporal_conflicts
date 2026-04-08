import pickle

file_path = "../CP_Output/sequences/mutable/mutable_Qwen2-7B-Instruct_persuasion_output.pkl"

with open(file_path, "rb") as f:
    try:
        pickle.load(f)
    except Exception as e:
        print("Error while loading:", e)
        print("\nPickle traceback info:\n")
        import traceback
        traceback.print_exc()

        # Try to peek at metadata (protocol, etc.)
        f.seek(0)
        raw = f.read(200)  # first 200 bytes
        print("\nFirst 200 bytes of pickle file:\n", raw)
