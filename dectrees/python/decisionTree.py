import monkdata as m

import dtree as d
from drawtree_qt5 import drawTree

print("Monk1")
t1 = d.buildTree(m.monk1, m.attributes)
print("E_train1:",1-d.check(t1, m.monk1))
print("E_test1",1-d.check(t1, m.monk1test))

print("Monk2")
t2 = d.buildTree(m.monk2, m.attributes)
print("E_train2:",1-d.check(t2, m.monk2))
print("E_test2",1-d.check(t2, m.monk2test))

print("Monk3")
t3 = d.buildTree(m.monk3, m.attributes)
print("E_train3:",1-d.check(t3, m.monk3))
print("E_test3",1-d.check(t3, m.monk3test))




