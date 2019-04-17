import os
def video_dataset_analysis(path_to_dataset):
    """This function takes as input a directory root dataset folder with the following structure:"
    dataset---
        ---class1 (must be a folder)
            ---video1 (must be a folder)
                ---frame1
                --frame2
                --- frame N
            ---video2 (must be a folder)
                ---frame1
                --frame2
                --- frame N
        ---class2 (must be a folder)
            ---video1 (must be a folder)
                ---frame1
                --frame2
                --- frame N
            ---video2 (must be a folder)
                ---frame1
                --frame2
                --- frame N
    Returns:
        clases: list of classes (class folder name)  eg. clases = [class1,class2]
        class_path: list of absolute paths to classes eg class_path = [absolute_path/class1, absolute_pàth/class2]
        files: python dictionary whose keys are the clases and items are class videos
                eg. files = {class1:[video1,video2,...,videoN]}
        files_path: Analogous to class_path
    """
    clases = [f for f in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset,f)) and f !='.' and f !='..']
    class_path = [os.path.join(path_to_dataset,f) for f in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset,f)) and f !='.' and f !='..']
    files = {}
    files_path = {}
    for clase,typ in zip(clases,class_path):
        files_ = [f for f in os.listdir(typ) if os.path.isdir(os.path.join(typ,f)) and f !='.' and f !='..']
        files_path_ = [os.path.join(typ,f) for f in os.listdir(typ) if os.path.isdir(os.path.join(typ,f)) and f !='.' and f !='..']
        files[clase]=files_
        files_path[clase] = files_path_
    return clases,class_path,files,files_path

def dataset_analysis(path_to_dataset):
    """This function takes as input a directory root dataset folder with the following structure:"
    dataset---
        ---class1 (must be a folder)
            ---file1 (must not to  be a folder)
            ---file2 (must not to be a folder)
        ---class2 (must be a folder)
            ---file1 (must not to be a folder)
            ---file2 (must not to be a folder)
    Returns:
        clases: list of classes (class folder name)  eg. clases = [class1,class2]
        class_path: list of absolute paths to classes eg class_path = [absolute_path/class1, absolute_pàth/class2]
        files: python dictionary whose keys are the clases and items are class videos
                eg. files = {class1:[video1,video2,...,videoN]}
        files_path: Analogous to class_path
    """
    clases = [f for f in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset,f)) and f !='.' and f !='..']
    class_path = [os.path.join(path_to_dataset,f) for f in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset,f)) and f !='.' and f !='..']
    files = {}
    files_path = {}
    for clase,typ in zip(clases,class_path):
        files_ = [f for f in os.listdir(typ)]
        files_path_ = [os.path.join(typ,f) for f in os.listdir(typ)]
        files[clase]=files_
        files_path[clase] = files_path_
    return clases,class_path,files,files_path

