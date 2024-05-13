import iso639

print("Fixing iso639 version") # Resolve iso639 and iso-639 conflict
with open(iso639.__file__, "a") as f:
    f.write("\n__version__ =\"0.1.4\"\n")
    print("Fixed")
