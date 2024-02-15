
# ZINC20 3D Tranches Downloader

[ZINC20](https://zinc20.docking.org/), [Files](https://files.docking.org/3D/)  

Physico-chemical property space in 3D is organized into 121 first-level tranches: **size** and **polarity**. Within each of these, we subdivide into a further four dimensions: **reactivity**, **purchasability**, **pH** and **charge**. This results in a six-dimensional space, as follows:

| Position | Description | Option | 
|--|--|--|
| 1 | molecular weight | A(<200)~K(500<) | 
| 2 | logP | A(<-1)~K(5<) | 
| 3 | reactivity | A=anodyne. B=Bother (e.g. chromophores) C=clean (but pains ok), E=mild reactivity ok, G=reactive ok, I = hot chemistry ok | 
| 4 | purchasability | A and B = in stock, C = in stock via agent, D = make on demand, E = boutique (expensive), F=annotated (not for sale) |
| 5 | pH range | R = ref (7.4), M = mid (near 7.4), L = low (around 6.4), H=high (around 8.4) | 
| 6 | net molecular charge | N = neutral, M = minus 1, L = minus 2 (or greater). O = plus 1, P = plus 2 (or greater) |

## Drug-like Dataset 

| Position | Description | Range | 
|--|--|--|
| 1 | molecular weight | < 500 | 
| 2 | logP | < 4 | 
| 3 | reactivity | A~E | 
| 4 | purchasability | A, B |
| 5 | pH range | R, M | 
| 6 | net molecular charge | N, M, O |

- Total Tranches : **3955**
- Total Promoters(molecule) : **8985308**

## ADDITION DATA

| Position | Description | Range | 
|--|--|--|
| 1 | molecular weight | A, J | 
| 2 | logP | < 4 | 
| 3 | reactivity | A~E | 
| 4 | purchasability | A, B, C |
| 5 | pH range | R, M | 
| 6 | net molecular charge | N, M, O |

- Total Promoters(molecule) : **2,908,326**