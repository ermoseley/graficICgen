def cmaps(cmap):
    from matplotlib.colors import LinearSegmentedColormap
    valid_cmaps = ['cmapkk1', 'cmapkk2', 'cmapkk3', 'cmapkk4', 
                'cmapkk5', 'cmapkk6', 'cmapkk7', 'cmapkk8', 'cmapkk9', 'cmapkk10']
    
    if cmap not in valid_cmaps:
        raise ValueError(f"Invalid colormap '{cmap}'. Valid options: {valid_cmaps}")
 
    if cmap=='cmapkk1':
        cmap1={
            'red': (
                (0.0,0.26,0.26),
                (.2,.32,.32),
                (0.35,.23,0.23),
                (.62,0.2,0.2),
                (.87,.63,.63),
                (1,.82,.82),
            ),
            'green': (
                (0.0,.0,0.0),
                (0.22,0.07,0.07),
                (.38,0.22,0.22),
                (.62,0.53,0.53),
                (.74,0.79,0.79),
                (1,.96,.96),
            ),
            'blue': (
                (0.0,0.24,0.24),
                (.2,0.36,0.36),
                (.36,.43,.43),
                (.62,.64,.64),
                (.67,.68,.68),
                (1,.85,0.85),
            )
        }
            
        cmapkk1= LinearSegmentedColormap('cmapkk1',cmap1)
        return cmapkk1
    elif cmap=='cmapkk2':

        cmap2={
            'red': (
                (0,0,0),
                (0.3,.04,.04),
                (0.4,.1,.1),
                (.5,.3,.3),
                (.65,.48,.48),
                (1,.94,.94),
            ),
            'green': (
                (0,.18,0.18),
                (.20,.27,.27),
                (.33,.27,.27),
                (.42,0.3,.3),
                (.63,.35,.35),
                (.72,.45,.45),
                (1,.85,.85),
            ),
            'blue': (
                (0.0,0.07,0.07),
                (.1,.17,.17),
                (.22,.27,.27),
                (.5,.52,.52),
                (.72,.7,.7),
                (1,.96,.96),
            )
        }    
            
        cmapkk2= LinearSegmentedColormap('cmapkk2',cmap2)
        return cmapkk2
    elif cmap=='cmapkk3':
        cmap3={
            'red': (
                (0.0,1,1),
                (.1, .69, .69),
                (.6, .14, .14),
                (1,0.02,0.02),
            ),
            'green': (
                (0.0,1,1),
                (0.1,0.79,0.79),
                (0.5,0.54,0.54),
                (.6, .5, .5),
                (1,.15,.15),
            ),
            'blue': (
                (0.0,1,1),
                (.11,0.87,0.87),
                (.45,0.67,0.67),
                (.65, .53, .53),
                (.85, .2, .2),
                (1,.05,0.05),
            )
        }
            
        cmapkk3= LinearSegmentedColormap('cmapkk3',cmap3)
        return cmapkk3
    elif cmap=='cmapkk4':

        cmap4={
            "red": (
                (0,.58,.58),
                (0.33,.4,0.4),
                (.5,0.2,0.2),
                (.87,.1,.1),
                (1,.15,.15),
            ),
            'green': (
        
                (0.,0.15,0.15),
                (.38,0.35,0.35),
                (.57,0.47,0.47),
                (.75,0.68,0.68),
                (1,.67,.67),
            ),
            'blue': (
                (0,0.42,0.42),
                (.3,.61,.61),
                (.5,.75,.75),
                (.75,.65,.65),
                (1,.35,0.35),
            )
        }
            
        cmapkk4=LinearSegmentedColormap('cmapkk4',cmap4)
        return cmapkk4
    elif cmap=='cmapkk5':
        cmap5={
            "red": (
                (0,0,0),
                (1,.85,.85),
            ),
            'green': (

                (0.,0,0),
                (1,.85,.85),
            ),
            'blue': (
                (0,0,0),
                (1,.85,0.85),
            )
        }

            
        cmapkk5=LinearSegmentedColormap('cmapkk5',cmap5)
        
        return cmapkk5
    elif cmap=='cmapkk6':
        cmap6={
            "red": (
                (0,.87,.87),
                (1,0.15,0.15),
                
            ),
            'green': (
                (0,.8,.8),
                (1.,0.08,0.08),
                
            ),
            'blue': (
                (0,.85,0.85),
                (1,0.1,0.1),
                
            )
        }

            
        cmapkk6=LinearSegmentedColormap('cmapkk6',cmap6)
        return cmapkk6
    elif cmap=='cmapkk7':
        cmap7={
            "red": (
                (0,.12,.12),
                (1,0.95,0.95),
                
            ),
            'green': (
                (0,0.05,0.05),
                (1.,0.9,0.9),
                
            ),
            'blue': (
                (0,0,0.05),
                (1,.8,.8),
                
            )
        }

            
        cmapkk7=LinearSegmentedColormap('cmapkk7',cmap7)
        return cmapkk7
    elif cmap=='cmapkk8':
        cmapkk8={
            'red': (
                (0.0,0.0,0.1),
                (.5,.8,.8),
                (1,.25,.25),
            ),
            'green': (
                (0.0,0.0,0.4),
                (.5,.8,.8),
                (.95,0,0),
                (1,0,0),
            ),
            'blue': (
                (0.0,0.6,.6),
                (.5,.8,.8),
                (1,.23,.23),
            )
        }
        cmapkk8=LinearSegmentedColormap('cmapkk8',cmapkk8)
        return cmapkk8
    elif cmap=='cmapkk9':
        cmapkk9={
            "red": (
                (0,.0,.0),
                (.45,.49,.49),
                (.75,.64,.64),
                (1,0.73,0.73),
                
            ),
            'green': (
                (0,0.0,0.0),
                (.4,0.15,0.15),
                (.6,0.35,0.35),
                (1.,0.87,0.87),
                
            ),
            'blue': (
                (0,0.01,0.01),
                (.2,0.21,0.21),
                (0.5,0.34,0.34),
                (0.7,0.42,0.42),
                (1,.7,.7),
                
            )
        }


            
        cmapkk9=LinearSegmentedColormap('cmapkk9',cmapkk9)
        
        return cmapkk9
    elif cmap=="cmapkk10":
        cmapkk10={
            'red': (
                (0.0,0.1,0.1),
                (.2,.22,.22),
                (0.35,.2,0.20),
                (.62,0.28,0.28),
                (.87,.63,.63),
                (1,.92,.92),
            ),
            'green': (
                (0.0,.0,0.0),
                (0.22,0.03,0.03),
                (.38,0.18,0.18),
                (.62,0.57,0.57),
                (.74,0.82,0.82),
                (1,.99,.99),
            ),
            'blue': (
                (0.0,0.06,0.06),
                (.2,0.27,0.27),
                (.36,.41,.41),
                (.62,.69,.69),
                (.68,.75,.75),
                (1,.93,0.89),
            )
        }
            
        cmapkk10=LinearSegmentedColormap('cmapkk10',cmapkk10)
        return cmapkk10