# tst point to line distance
import numpy as np

def dpt2line(p1, p2, pt):
    x0 = pt[0]  # alg source: wikipedia "distance point to line"
    y0 = pt[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    num = np.abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    den = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return num/den

tp1 = (20,20)       # a line at 45deg
tp2 = (10,10)

eps = 0.000002

def test_sh (pt, correct):
    assert np.abs(correct - dpt2line(tp1,tp2,pt))< eps, 'failed'
    
test_sh((4,4),0.0)
r = np.sqrt(25+25)
test_sh((0,10), r)
test_sh((-1,-1), 0.0)
r = np.sqrt(2)*0.5
test_sh((500,499), r)
test_sh((500,501), r)
test_sh((-500,-501), r)

