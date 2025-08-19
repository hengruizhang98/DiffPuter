# Baselines 

## Optimal transport baseline: MOT and TDM
```python
python OT_benchmark.py  --mask_type [MCAR/MAR/MNAR] 
```

## GRAPE baseline
```bash
cd GRAPE
```

```python
python GRAPE_benchmark.py  --mask_type [MCAR/MAR/MNAR] 
```

## IGRM baseline
```bash
cd IGRM
```

```python
python IGRM_benchmark.py  --mask_type [MCAR/MAR/MNAR] 
```

## Remasker baseline
```bash
cd remasker
```

```python
python remasker_benchmark.py --mask_type [MCAR/MAR/MNAR]
```

## Hyperimpute and traditional methods baseline
```bash
cd hyperimpute
```

```python
python hyperimpute_benchmark.py --mask_type [MCAR/MAR/MNAR]
```

## MissDiff
```bash
cd Missdiff_SDE
```

```python
python Missdiff_benchmark.py --mask_type  [MCAR/MAR/MNAR]
```

## MCFlow

```bash
cd MCFlow
```

```python
python MCFlow_benchmark.py --mask_type [MCAR/MAR/MNAR]
```

## TabCSDI

```bash
cd TabCSDI
```

```python
python csdi_benchmark.py 
```
