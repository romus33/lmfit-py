import struct
import copy
from ctypes import *
import h5py
import os
import numpy as np
import platform
from termcolor import colored
import time as tm
import progressbar as pb
from scipy.stats.stats import pearsonr

os.system('color')

# check platform
if platform.uname()[0] == "Windows":
    name_dll = "convolution.dll"
else:
    name_dll = "convolution.so"
myclib = cdll.LoadLibrary(name_dll)



class ReadWrite5(object):
    # Class can operate with ir5 and h5 files
    def __init__(self):
        self.lam_als = 1e7
        self.p_als = 0.01
        self.x1=[]
        self.x2=[]
        self.y1=[]
        self.y2=[]
        self.__pBar__ = pb.ProgressBar(tm.time())


    def pairwise_correlation(self, A, B):
        am = A - np.mean(A, axis=0, keepdims=True)
        bm = B - np.mean(B, axis=0, keepdims=True)
        return am.T @ bm /  (np.sqrt(
            np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
            np.sum(bm**2, axis=0, keepdims=True)))

    @staticmethod
    # Change dataset in h5 database
    # fname is file name, group is groups name where dataset is located; cnt is number of the dataset;
    # data is array of y values of the dataset
    # attribute is dictionary of dataset attributes. attributes{"xmin":float32,"xmax":float32,"npoints:int,"name":str,"formula":str,"composition":str,"type":str}
    # "xmin" is min value on x axis
    # "xmax" is max value on x-axis
    # "npoints" is number of point in spectra (len (data))
    # "name" is mineral name and locality
    # formula is chemical formula of the mineral
    # composition is chemical composition of the mineral
    # type is type of spectra ftir, raman etc
    ###########################################
    def changedb_h5(fname, group="uncorrected", cnt=0, data=None, attribute=None):
        with h5py.File(fname, 'a') as f:
            gr = f.get(group)
            if data:
                gr[cnt] = data
            try:
                for key in attribute:
                    gr[cnt].attr[key] = attribute[key]
            except Exception as error:
                print(error)

    ############################
    # Add dataset in h5 database
    # fname is file name,
    # group is groups name where dataset is located
    # spectrum is array of [x,y] values of the dataset
    # "name" is mineral name and locality
    # formula is chemical formula of the mineral
    # composition is chemical composition of the mineral
    # type is type of spectra ftir, raman etc
    ###########################################
    @staticmethod
    def addtodb_h5(fname, spectrum, group='uncorrected', name="", formula="", composition="", type_m="ftir"):

        with h5py.File(fname, 'a') as f:
            if not f.__contains__(group):
                gr = f.create_group(group)
            gr = f.get(group)
            ls = list(gr.keys())
            cnt = len(ls)
            x_= np.array(spectrum[0])
            data_ = np.array(spectrum[1])

            d = gr.create_dataset(str(cnt + 1), data_.shape, data=data_, dtype='f', compression="gzip",
                                  compression_opts=9)
            d.attrs['name'] = name
            d.attrs["xmin"] = x_.min()
            d.attrs["xmax"] = x_.max()
            d.attrs["npoints"] = len(x_)
            d.attrs["formula"] = formula
            d.attrs["composition"] = composition
            d.attrs["type"] = type_m
            gr = f.get(group)
            ls = list(gr.keys())
            print(len(ls))

    # Read h5 file to dictionary

    def readh5(self, fname, group='uncorrected'):
        db = {}
        print('Reading ' + colored(fname, 'blue') + ' file:')
        with h5py.File(fname, 'r') as f:

            count_i = 0
            gr = f.get(group)
            ls = list(gr.keys())
            length=len(ls)
            self.__pBar__.printProgressBar(0, length, prefix='Progress:', suffix='', length=50)
            for key in ls:
                a = gr.get(key)
                df = {"xmin": a.attrs["xmin"], "xmax": a.attrs["xmax"], "npoints": a.attrs["npoints"],
                      "ydata": np.array(a), "name": str(a.attrs["name"]), "formula": str(a.attrs["formula"]),
                      "type": str(a.attrs["type"]), "composition": str(a.attrs["composition"])}
                db[key] = df
                count_i = count_i+1
                self.__pBar__.printProgressBar(count_i, length, prefix='Progress:', suffix='', length=50)

        return db

    # Read all datasets to dictionary {[x, y, name]}. Key is number of mineral in database
    @staticmethod
    def readh5_all(db_):
            db={}
            for key, a in db_.items():
                #print(key, ' ', a)
                x = np.linspace(a["xmin"], a["xmax"], a["npoints"])
                db[key] = [x, a['ydata'], str(a['name'])]

                # spectra[cnt] = self.fun_spectra(xmin=xmin, xmax=xmax, npoints=npoints, ydata=ydata)
            return db

    @staticmethod
    def fun_filter_h5(db, line):
        return [db[i] for i in db if line in db[i]["name"]]

    # baseline substraction
    def __baseline_als_dll__(self, y):
        """Baseline reconstruct using ALS algorithm using external dll library convolution.dll
            https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
            Asymmetric Least Squares Smoothing
            niter - Maximum number of iterations
            lam - 2nd derivative constraint
            p - Weighting of positive residuals
        
        """
        # print('fff')
        y_arr = y
        # print(y_arr)
        lam = self.lam_als
        p = self.p_als
        niter = 10
        py_ALS = myclib.ALS
        py_ALS.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_double, c_double, c_int]
        py_ALS.restype = None
        inputarray = (c_double * len(y_arr))(*y_arr)
        outputarray = (c_double * len(y_arr))()
        N = c_int(len(y_arr))
        lam_dll = c_double(lam)
        p_dll = c_double(p)
        py_ALS(inputarray, outputarray, N, lam_dll, p_dll, niter)

        return np.array(outputarray)

    # convert ir5 to h5
    def h5convert(self, fname, db, substraction=False):

        with h5py.File(fname, 'w') as f:
            group_data = f.create_group('uncorrected')
            # group_data2=f.create_group('corrected')
            nrecords = db["nrecords"]

            for cnt in range(nrecords):
                # x = np.linspace(db[cnt]["xmin"], db[cnt]["xmax"], db[cnt]["npoints"])
                y = np.array(db[cnt]["ydata"])
                if substraction:
                    y = y - self.__baseline_als_dll__(y)
                    print('Substracted')
                name = str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00")
                arr = np.array(y)
                d = group_data.create_dataset(str(cnt), arr.shape, data=arr, dtype='f', compression="gzip",
                                              compression_opts=9)
                d.attrs['name'] = name
                # print(cnt)
                # print(db[cnt]["com2"])
                d.attrs['formula'] = str(db[cnt]["com2"])
                d.attrs['composition'] = ""
                d.attrs['xmin'] = db[cnt]["xmin"]
                d.attrs['xmax'] = db[cnt]["xmax"]
                d.attrs['npoints'] = db[cnt]["npoints"]
                d.attrs['type'] = 'ftir'

    @staticmethod
    def fun_spectra(xmin, xmax, npoints, ydata):
        step = (xmax - xmin) / (npoints - 1)
        xarr = np.zeros(npoints)
        yarr = np.zeros(npoints)
        for i, j in enumerate(ydata):
            xarr[i] = xmin + i * step
            yarr[i] = j
        return xarr, yarr

    # Read ir5 database to dictionary
    def readir5(self, fname):
        """
        
        """
        # p = 0
        with open(fname, 'rb') as f:
            db = {}
            spectra = {}
            content = f.read()
            db["u1"] = content[0:8]
            db["nrecords"] = struct.unpack("<H", content[8:10])[0]
            # print(db["nrecords"])
            db["u2"] = content[10:14]  # zeroes
            db["fnamelen"] = struct.unpack("<H", content[14:16])[0]
            fl = db["fnamelen"]
            db["db_fname"] = content[16:16 + fl]
            db["u3"] = content[16 + fl:16 + fl + 2]
            # print(db["u3"])
            nrecords = db["nrecords"]
            p = 18 + fl
            for cnt in range(nrecords):
                df = {}
                reclen1, bpy, svz, npoints, xmin, xmax, time, reclen2, comlen1 = struct.unpack("<HbbHffiHH",
                                                                                               content[p:p + 22])
                # print(comlen1)
                p = p + 22
                com1 = content[p:p + comlen1]
                p = p + comlen1
                comlen2 = struct.unpack("<H", content[p: p + 2])[0]
                p = p + 2
                com2 = content[p:p + comlen2]
                p = p + comlen2
                comlen3 = struct.unpack("<H", content[p: p + 2])[0]
                # print(comlen1)
                p = p + 2
                com3 = content[p:p + comlen3]
                p = p + comlen3
                ydata = [(1 - i / 65535) * 100 for i in
                         struct.unpack("<" + "H" * npoints, content[p: p + npoints * bpy:])]
                p = p + npoints * bpy + 2

                com4 = content[p - 2: p]  # finalization 02 00 ?
                df["reclen1"] = reclen1  # length of record. true length of record = reclen1 + 4
                df["reclen2"] = reclen2  # equal to reclen1
                df["bpy"] = bpy  # bytes per data point
                df["svz"] = svz  # ?
                df["npoints"] = npoints  # points in yarray
                df["xmin"] = xmin
                df["xmax"] = xmax
                df["time"] = time  # ?

                df["comlen1"] = comlen1
                df["comlen2"] = comlen2
                df["comlen3"] = comlen3
                df["com1"] = com1  # mineral, zeroes
                df["com2"] = com2  # formula, zeroes
                df["com3"] = com3  # date, time, unknown, [unknown / MCT], zeroes
                df["ydata"] = ydata
                df["com4"] = com4  # finalization 02 00 ?
                db[cnt] = df
                spectra[cnt] = self.fun_spectra(xmin=xmin, xmax=xmax, npoints=npoints, ydata=ydata)
                # print(cnt," ", df["com1"], " ", df["com4"])
            # if not p - 2 == len(content):
            #    print("file is not read up to end", file=sys.stderr)
        return db

    # Write db dictionary to ir5 database
    @staticmethod
    def writeir5(db, fname):
        """
        structure of database
        unknown [0:8], nrecords [8:10], unknown (zeroes) [10:14]
        length of db_name [14:16], db_name, unknown 2 bytes
        """
        fout = open(fname, 'wb')
        fout.write(db["u1"])
        fout.write(struct.pack("<H", db["nrecords"]))
        fout.write(db["u2"])
        fout.write(struct.pack("<H", db["fnamelen"]))
        fout.write(db["db_fname"])
        fout.write(db["u3"])

        for cnt in range(db["nrecords"]):
            """
                structure of record
                length of record 2 (i), bytes per y-point 1 (i), svz 1 (i),
                npoints 2 (i), xmin 4 (f), xmax 4 (f), time 4 (i), length of record 2(i),
                length of comment 1, comment1,
                length of comment 2, comment2,
                length of comment 3, comment3, ydata, comment4 "02 00"
                """
            df = db[cnt]
            fout.write(struct.pack("<H", df["reclen1"]))
            fout.write(struct.pack("<b", df["bpy"]))
            fout.write(struct.pack("<b", df["svz"]))
            fout.write(struct.pack("<H", df["npoints"]))
            fout.write(struct.pack("<f", df["xmin"]))
            fout.write(struct.pack("<f", df["xmax"]))
            fout.write(struct.pack("<i", df["time"]))
            fout.write(struct.pack("<H", df["reclen2"]))

            fout.write(struct.pack("<H", df["comlen1"]))

            fout.write(df["com1"])
            fout.write(struct.pack("<H", df["comlen2"]))
            fout.write(df["com2"])
            fout.write(struct.pack("<H", df["comlen3"]))
            fout.write(df["com3"])
            for y in df["ydata"]:
                fout.write(struct.pack("<H", int((1 - y / 100) * 65535)))
            fout.write(df["com4"])

    # Add record from db dictionary to ir5 database
    @staticmethod
    def add_record_ir5(record, db):
        """ add record at end of db, increase nrecords"""
        ln = copy.deepcopy(db["nrecords"])
        # print(ln)
        db[ln] = record
        db[ln - 1]["com4"] = db[ln - 2]["com4"]
        db["nrecords"] += 1
        print("The record has been added")

    # delete record from ir5 database
    @staticmethod
    def delete_record_ir5(db, _n):
        """decrease nrecords, remove and return the n-th record"""

        db[_n] -= 1
        return db.pop(_n)

    # change db name of ir5 database
    @staticmethod
    def db_change_prop_ir5(db, prop, value):
        """change db_fname"""
        if prop == "db_fname":
            db[prop] = bytes(value, "UTF-8")
            db["fnamelen"] = len(value)

    # change property of ir5 database
    @staticmethod
    def entry_change_prop_ir5(db, _n, prop, value):
        if prop == "spectrum":
            """value = [xmin, xmax, ydata]"""
            db[_n]["xmin"], db[_n]["xmax"], db[_n]["ydata"] = value
            new_npoints = len(db[_n]["ydata"])
            db[_n]["reclen1"] += (new_npoints - db[_n]["npoints"]) * db[_n]["bpy"]
            db[_n]["reclen2"] = db[_n]["reclen1"]
            db[_n]["npoints"] = new_npoints
        elif prop == "mineral":
            """value = mineral_name """
            old_comlen1 = db[_n]["comlen1"]
            new_comlen1 = len(value)
            db[_n]["comlen1"] = new_comlen1
            db[_n]["com1"] = bytes(value, "UTF-8")  # possibly, zero bytes at the end are required
            db[_n]["reclen1"] += new_comlen1 - old_comlen1
            db[_n]["reclen2"] = db[_n]["reclen1"]
        elif prop == "formula":
            """value = chem_formula """
            old_comlen2 = db[_n]["comlen2"]
            new_comlen2 = len(value)
            db[_n]["comlen2"] = new_comlen2
            db[_n]["com2"] = bytes(value, "UTF-8")  # possibly, zero bytes at the end are required
            db[_n]["reclen1"] += new_comlen2 - old_comlen2
            db[_n]["reclen2"] = db[_n]["reclen1"]
        else:
            print("changing property not implemented")

    # get dictionary of multiple spectra from ir5
    @staticmethod
    def get_multiple_spectra_ir5(db, numbers):
        spectra = {}

        for cnt in numbers:
            # spectra[str(cnt)] = [xmin, xmax, step,self.__fun_spectra__(npoints,ydata),str(com1.decode("UTF-8")).rstrip("\x00"),npoints]
            spectra[str(cnt)] = [db[cnt]["xmin"], db[cnt]["xmax"], db[cnt]["npoints"], db[cnt]["ydata"],
                                 str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00")]
        return spectra

    #
    @staticmethod
    def getspectra_ir5(db):
        spectra = {}
        nrecords = db["nrecords"]
        for cnt in range(nrecords):
            spectra[cnt] = db[cnt]["ydata"]
        return spectra

    # get spectra from database ir 5 to dict[x,y, name]
    @staticmethod
    def getspectra_all_ir5(db):
        spectra = {}
        nrecords = db["nrecords"]
        for cnt in range(nrecords):
            x = np.linspace(db[cnt]["xmin"], db[cnt]["xmax"], db[cnt]["npoints"])
            spectra[str(cnt)] = [x, db[cnt]["ydata"], str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00")]
        return spectra

    @staticmethod
    def fun_filter_ir5(db, line):
        return [db[i] for i in range(db["nrecords"]) if line in db[i]["com1"]]

    # get list of minerals from ir5 db
    @staticmethod
    def get_list_ir5(db, fname):
        nrecords = db["nrecords"]
        for cnt in range(nrecords):
            print(str(cnt) + '. ' + str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00"))
            with open(fname, 'a') as fl:
                fl.write(str(cnt) + '.\t' + str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00") + '\n')
                print(str(cnt) + '.\t' + str(db[cnt]["com1"].decode("UTF-8")).rstrip("\x00"))

    # find phase close to input [x,y] spectrumin ir5
    @staticmethod
    def findphase_ir5(x, y, db, r_ref=0.95):
        nrecords = db["nrecords"]
        tmp_arr = np.zeros(nrecords)
        for cnt in range(nrecords):
            dbY = db[cnt]["ydata"]
            dbX = np.linspace(db[cnt]["xmin"], db[cnt]["xmax"], db[cnt]["npoints"])
            dbY = np.interp(x, dbX, dbY)
            r = np.dot(dbY, y) / (np.linalg.norm(dbY) * np.linalg.norm(y))
            tmp_arr[cnt] = r
        return filled(tmp_arr, r_ref)

    # find phase close to input [x,y] spectrumin h5


    def findphase_h5(self, x, y, db, r_ref=0.95):
        nrecords = len(db)
        tmp_arr = np.zeros(nrecords)

        for cnt in range(nrecords):
            if cnt==0:
                if "0" in db:
                    cnt=0
                else:
                    cnt=1
            dbX = np.linspace(db[str(cnt)]["xmin"], db[str(cnt)]["xmax"], db[str(cnt)]["npoints"])
            #print(dbX)
            #print(x)
            dbY = np.array(db[str(cnt)]["ydata"])
            dbY = np.interp(x, dbX, dbY)
            dbY=dbY/max(dbY)
            # dbY = dbY-self.__baseline_als_dll__(dbY)
            r = np.dot(dbY, y) / (np.linalg.norm(dbY) * np.linalg.norm(y))
            tmp_arr[cnt] = r
        return filled(tmp_arr, r_ref)
    def find_phase_in(self, x, y, dbname=None, r_ref=0.8):
        if dbname is not None:
            y=np.array(y) 
            x=np.array(x)
            f = h5py.File(dbname, 'r')
            count_i = 0
            group='uncorrected'
            gr = f.get(group)
            ls = list(gr.keys())
            length=len(ls)
            self.__pBar__.printProgressBar(0, length, prefix='Progress:', suffix='', length=50)
            found_phases_=[]
            for key in ls:
                a = gr.get(key)
                dbX = np.array(a[0])
                dbY = np.array(a[1])
                dbY_s = dbY - np.array(a[2])
                dbY_s = np.interp(x, dbX, dbY_s)
                r_ = self.pairwise_correlation(y, dbY_s)[0]
                dbY = np.interp(x, dbX, dbY)
                r = self.pairwise_correlation(y, dbY)[0]
                if r>r_:
                    if (r>=r_ref):
                        #print('!!!!!')
                        found_phases_.append({"key": key, "r": r, "name":  a.attrs["name"], "x": x, "y": dbY})
                else:
                    if (r_>=r_ref):
                        #print('??????')
                        found_phases_.append({"key": key, "r": r_, "name":  a.attrs["name"], "x": x, "y": dbY})
                count_i = count_i+1
                self.__pBar__.printProgressBar(count_i, length, prefix='Progress:', suffix='', length=50)

            return found_phases_
        else:
            raise ValueError(f'Empty db name')     
        