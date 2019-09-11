import os

class CSVLog():
    def __init__(self, filename, mode='x', index=None, keep_file_open=False):
        """
        Very basic CSV logger.
        :param filename: File path to log to
        :param mode: Single character with one of the following values
            'a' - append to the end of the log
            'x' - create new file, fail if exists
            'w' - create new file, truncate existing.
        :param index: Name of column name to use as an index (similar to pandas Dataframes)
        :param autoflush: Boolen, whether to flush after every call, default is True.

        The resulting object is callable using keywork=value arguments for fieldname=value in the log
        >>> log = CSVLog('log.csv', index='epoch')
        >>> log(epoch=0, foo='yes', bar=False, loss=0.9)
        >>> log(epoch=1, foo='yes', bar=True, loss=0.8)
        >>> log(foo='yes', loss=0.7, bar=False, epoch=2)
        >>> log(epoch=3, foo='no', bar=False, loss=0.1)
        >>> log.close()
        >>> %cat log.csv
            epoch,foo,bar,loss
            0,'yes',False,0.9
            1,'yes',True,0.8
            2,'yes',False,0.7
            3,'no',False,0.1

        """

        if mode in ('x', 'w'):
            self._fp = open(filename, mode=mode+'+')
        elif mode == 'a':
            if os.path.exists(filename):
                self._fp = open(filename, mode='r+')
            else:
                mode = 'x'
                self._fp = open(filename, mode='w+')
        else:
            raise ValueError(f'invalid mode: {mode} - must be one of a,x,w')

        self._fname = filename
        self._keep_file_open = keep_file_open

        if mode=='a':
            self._fp.seek(0,0)
            header = self._fp.readline()
            if len(header.strip()) == 0:
                header = []
            else:
                header = header.strip('\n').split(',')
            self._columns = set(header)
            self._columns_order = header
            self._fmt_str = ",".join("{" + c + "!s}" for c in self._columns_order)
            self._fp.seek(0,2)

        else:
            if index is not None:
                self._columns = {index}
                self._columns_order = [index]
                self._fmt_str = "{" + index + "!s}"

            else:
                self._columns = set()
                self._columns_order = []
                self._fmt_str = ""

            self._fp.write("\n")
            self._fp.flush()

        if not self._keep_file_open:
            self._fp.close()


    def __call__(self, **kwargs):
        kw = {k: "" for k in self._columns}
        kw.update(kwargs)

        for key in kw:
            if key not in self._columns:
                self._columns.add(key)
                self._columns_order.append(key)
                if len(self._fmt_str) > 0:
                    self._fmt_str += ",{" + key + "!s}"
                else:
                    self._fmt_str = "{" + key + "!s}"

        if not self._keep_file_open:
            self._fp = open(self._fname, 'r+')
            self._fp.seek(0, 2)

        self._fp.write(self._fmt_str.format_map(kw) + "\n")
        self._fp.flush()

        if not self._keep_file_open:
            self._fp.close()

    def update_header(self):
        """Update the header in the file, add empty fields if required.  The entire file is re-written."""
        if not self._keep_file_open:
            self._fp = open(self._fname, 'r+')

        self._fp.seek(0,0)
        old_header = self._fp.readline()
        new_header = ",".join(self._columns_order) + "\n"
        if old_header == new_header:
            return
        lines = self._fp.readlines()

        num_cols = len(self._columns_order)
        for i in range(len(lines)):
            lines[i] = lines[i][:-1] + "," * (num_cols - lines[i].count(",") - 1) + "\n"

        self._fp.seek(0,0)
        self._fp.write(new_header)
        self._fp.writelines(lines)
        self._fp.truncate(self._fp.tell())
        self._fp.flush()

        if not self._keep_file_open:
            self._fp.close()

    def close(self):
        if hasattr(self, "_fp") and self._fp is not None:
            self.update_header()
            if  self._keep_file_open:
                self._fp.close()
                self._fp = None


