import os
import subprocess


folder_path = os.path.expanduser('~/Desktop/Encrypted') 

output_dmg_path = os.path.expanduser('~/Desktop/Encrypted.dmg')

volume_name = 'Encrypted'

disk_image_size = '500m'

command_create = f'hdiutil create -volname "{volume_name}" -srcfolder "{folder_path}" ' \
                 f'-encryption AES-256 -format UDRW -size {disk_image_size} "{output_dmg_path}"'

# Command to create the disk image
try:
    result_create = subprocess.run(command_create, shell=True, check=True, capture_output=True, text=True)
    print("Encrypted disk image created successfully.")

    # Command to mount the disk image
    command_mount = f'hdiutil attach "{output_dmg_path}"'
    result_mount = subprocess.run(command_mount, shell=True, check=True, capture_output=True, text=True)
    print("Disk image mounted successfully.")

    # Optional: Open the mounted volume in Finder
    mount_point_path = f'/Volumes/{volume_name}'
    subprocess.run(f'open "{mount_point_path}"', shell=True, check=True)

except subprocess.CalledProcessError as e:
    print(f"Error: {e.stderr}")