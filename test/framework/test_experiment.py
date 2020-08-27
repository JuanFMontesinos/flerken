import unittest
from flerken.framework.experiment import FileManager, experiment_cfg, Arxiv, arxiv_cfg
import shutil
import os


class TestFileManager(unittest.TestCase):
    def setUp(self) -> None:
        self.real_path = './filemanager_test'
        self.fake_path = './filemanager_test_unexistent'
        if not os.path.exists(self.real_path):
            os.mkdir(self.real_path)

    def tearDown(self):
        if os.path.exists(self.fake_path):
            shutil.rmtree(self.fake_path)
        if os.path.exists(self.real_path):
            shutil.rmtree(self.real_path)

    def test_init_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            filemanager = FileManager(self.real_path)

    def test_init_resume(self):
        json = experiment_cfg()["cfg_files_dicttype"]()
        json.write(os.path.join(self.real_path, experiment_cfg()["internal_cfg_filename"]))
        self.assertTrue(os.path.exists(os.path.join(self.real_path, experiment_cfg()["internal_cfg_filename"])))
        filemanager = FileManager(self.real_path)
        print(filemanager.resume)
        self.assertTrue(filemanager.resume)

    def test_init_not_resume(self):
        if os.path.exists(self.fake_path):
            shutil.rmtree(self.fake_path)
        filemanager = FileManager(self.fake_path)
        self.assertTrue(os.path.exists(self.fake_path))
        self.assertTrue(os.path.join(filemanager.workdir, experiment_cfg()["metadata_foldername"]))
        self.assertTrue(os.path.exists(os.path.join(self.fake_path, experiment_cfg()["internal_cfg_filename"])))

        if os.path.exists(self.fake_path):
            shutil.rmtree(self.fake_path)

    def test_add_cfg_files(self):
        json = experiment_cfg()["cfg_files_dicttype"]()
        json.write(os.path.join(self.real_path, experiment_cfg()["internal_cfg_filename"]))
        filemanager = FileManager(self.real_path)
        cfg = {'ivan': 18, 'jamon': True}
        filemanager.add_cfg('test_cfg', cfg)
        cfg_base = experiment_cfg()["cfg_files_dicttype"](cfg)
        self.assertTrue(isinstance(filemanager.cfg_files, experiment_cfg()["cfg_files_dicttype"]))

    def test_sys_info(self):
        json = experiment_cfg()["cfg_files_dicttype"]()
        json.write(os.path.join(self.real_path, experiment_cfg()["internal_cfg_filename"]))
        filemanager = FileManager(self.real_path)
        print(filemanager.get_sys_info())

    def test_write_cfg(self):
        filemanager = FileManager(self.fake_path)
        cfg = {'jose ': 18, 'jamon': True}
        filemanager.add_cfg('test', cfg)
        filemanager.write_cfg()
        self.assertTrue(os.path.exists(os.path.join(self.fake_path,
                                                    experiment_cfg()["metadata_foldername"],
                                                    str(filemanager._internal_cfg['version']))))
        self.assertTrue(os.path.exists(os.path.join(filemanager.metadata_dir, "sysinfo.txt")))
        self.assertTrue(os.path.exists(os.path.join(filemanager.metadata_dir, "test.json")))
        self.assertTrue(os.path.exists(os.path.join(filemanager.metadata_dir, ".seed.json")))

    def test_summary_writer(self):
        filemanager = FileManager(self.fake_path)
        filemanager._set_writer()
        self.assertTrue(os.path.exists(os.path.join(filemanager.workdir, 'tensorboard')))
        self.assertTrue(len(os.listdir(os.path.join(filemanager.workdir, 'tensorboard'))) > 0)


class TestArxiv(unittest.TestCase):
    def setUp(self) -> None:
        self.real_path = './arxiv_test'
        self.fake_path = './arxivb_test_unexistent'
        if not os.path.exists(self.real_path):
            os.mkdir(self.real_path)

    def tearDown(self):
        if os.path.exists(self.fake_path):
            shutil.rmtree(self.fake_path)
        if os.path.exists(self.real_path):
            shutil.rmtree(self.real_path)

    def test_init(self):
        arxiv = Arxiv(self.fake_path)
        self.assertTrue(os.path.exists(os.path.join(arxiv.dir, arxiv_cfg()["database_name"])))
