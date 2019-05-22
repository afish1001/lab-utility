import os, zipfile


def make_zip(source_dir, output_filename): # 打包目录为zip文件（未压缩）
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dir_names, file_names in os.walk(source_dir):
        for file_name in file_names:
            path_file = os.path.join(parent, file_name)
            arc_name = path_file[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(path_file, arc_name)
    zipf.close()


def unzip_file(source_dir, output_dir): # 解压缩
    zipf = zipfile.ZipFile(source_dir)
    zipf.extractall(output_dir)
    zipf.close()
