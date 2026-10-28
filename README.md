# Лабораторная работа №3. Метод сопряжённых градиентов

Две MPI-реализации метода сопряжённых градиентов для задачи наименьших квадратов: с распределёнными и с полными рабочими векторами.

Отчёт, формулы, измерения, графики и дополнительные эксперименты находятся в [report.md](report.md).

## Запуск

Нужны Python и MPI. Код использует стандартные операции mpi4py. Для локальной проверки на Windows использован Intel MPI; в Linux можно использовать Open MPI или MPICH.

```bash
python -m pip install -r requirements.txt
mpiexec -n 4 python src/cg_full.py
mpiexec -n 4 python src/cg_simple.py
```

В `data/` находится система `1000 × 500`, в `data/uneven/` — система `7 × 5`. Файл `in.dat` содержит сначала `N`, затем `M`. Вектор правой части читается из `bData.dat`, известное решение для проверки генератора записано в `xTrue.dat`.

```bash
mpiexec -n 4 python src/cg_full.py --data-dir data/uneven
mpiexec -n 4 python src/cg_simple.py --alpha 0.0001 --tol 1e-10
```

Обе программы сравнивают ответ с `numpy.linalg.lstsq`. Результат записывается в `Results_full.dat` или `Results_simple.dat` в текущем каталоге. Выводится время только итерационного цикла. При недостижении допуска программа сообщает причину и завершается с ненулевым кодом.

## Проверки и эксперименты

```bash
python -m pip install -r requirements.txt
python scripts/benchmark.py
python scripts/studies.py
```

`mpiexec` должен быть в `PATH`; для `benchmark.py` можно вместо этого задать переменную `MPIEXEC`. В Windows MPI можно установить внутри виртуального окружения командой `python -m pip install impi-rt`.

`benchmark.py` выполняет замеры на 1, 2, 4 и 8 процессах. `studies.py` запускается обычной командой Python и использует `MPI.COMM_SELF` для опытов по регуляризации и остановке. Для воспроизведения этих опытов установите `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` и `MKL_NUM_THREADS` в `1` до запуска Python. `benchmark.py` задаёт эти значения для дочерних процессов сам.

Повторный запуск скриптов обновляет таблицы и графики в `report/`; после него числовые значения и выводы в `report.md` нужно сверить с новыми результатами.
