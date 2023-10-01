import NewHomeFunGame2 as G

x = G.PrimitiveValue(G.matrix_to_str(
    [[None, None, None, 'X'],
     ["O", 'O', None, None],
     [None, "X", 'X', None]]
    ) 
)
print(x)

x = G.PrimitiveValue("XOXXXOOXX")
print(x)


x = G.PrimitiveValue(G.matrix_to_str(
    [[None, 'X', None, 'X'],
     [None, 'O', None, None],
     [None, 'X', 'O', None]]
    ) 
)
print(x)

x = G.PrimitiveValue("NXXOOXXNONOX")
print(x)

