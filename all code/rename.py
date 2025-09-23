import os

# Folder containing your images
folder = r"valid"

# Starting number
start_num = 4052   # change this to whatever number you want (e.g. 101)

# List files (only images)
files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
files.sort()  # alphabetical order (you can also sort by time if needed)

for idx, filename in enumerate(files, start=start_num):
    ext = os.path.splitext(filename)[1]  # keep original extension
    new_name = f"{idx}{ext}"  # e.g., 1.jpg, 2.png, 3.jpeg
    
    src = os.path.join(folder, filename)
    dst = os.path.join(folder, new_name)
    
    os.rename(src, dst)
    print(f"{filename} -> {new_name}")

print("✅ Renaming done!")
