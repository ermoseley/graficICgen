This is the github repository for my new grafic IC generation scripts.
I grew quite tired of using MUSIC for something that didn't need it, 
so I rewrote it all as just python scripts. Now, there's no need to deal with
f2py compilation or anything like that; everything's just in python. 

It should be fairly clear how to use the code. By default, if you simply write (e.g.):
./generateIC 8

you'll generate initial conditions for a uniform box with a resolution of 2^8 = 256^3. 

Within generateIC, you can also specify the magnetic field strength. As well, if you want to modify the densities, velocities, etc.,
you can go into generate.py and make your own custom version of it. As it stands, the code just generates uniform ICs. 

Happy IC generation!
