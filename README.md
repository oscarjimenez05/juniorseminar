# Commands
## Windows

```shell
C:\Users\oscar\AppData\Local\Programs\Python\Python311\python.exe setup.py build_ext --inplace
```

## Linux

```shell
source venv/bin/activate
```

```shell
python3 setup.py build_ext --inplace
```

```shell
python3 dieharder_interface.py p | dieharder -g 200 -d <ID> > out.txt
```

```shell
python3 dieharder_interface.py f > out.bin
```

```shell
dieharder -g 201 -f in.bin -d <ID> > out.txt 2>&1
```

```bash
xxd -b file.bin | head -n 20
```

---

```shell
gcc test_from_pipe.c -o test_from_pipe \
  -I/usr/local/include \
  -L/usr/local/lib \
  -ltestu01 -lprobdist -lmylib -lm
```

```shell
python3 dieharder_interface.py p | ./test_from_pipe > results.txt
```

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```
