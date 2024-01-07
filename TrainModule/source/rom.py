SPLIT_POINT_INFO = {
    "player": (None         , None, False),
    "dn53"  : ([36, 61,  74],   75, False),
    "alt1"  : ([75, 90, 105],   76, True ),
}

GLOBAL_PARAMS = {                       # Offset,     Scale,       IP, FR,  FramesToBeEncoded
    "Traffic"                           : (-186.9549408, 354.9010315, 32, 30,   33),
    "Kimono"                            : (-169.5782928, 341.3738098, 32, 24,   33),
    "ParkScene"                         : (-211.3495789, 366.0599670, 32, 24,   33),
    "Cactus"                            : (-210.4352722, 453.4030914, 64, 50,   97),
    "BasketballDrive"                   : (-227.4010925, 409.1360931, 64, 50,   97),
    "BQTerrace"                         : (-186.9790955, 346.8961487, 64, 60,  129),
    "BasketballDrill"                   : (-198.3319702, 389.4044037, 64, 50,   97),
    "BQMall"                            : (-236.8316040, 461.4279022, 64, 60,  129),
    "PartyScene"                        : (-192.5734558, 353.2027130, 64, 50,   97),
    "RaceHorsesC"                       : (-279.9704895, 452.3226624, 32, 30,   65),
    "BasketballPass"                    : (-226.6271210, 408.6202850, 64, 50,   97),
    "BQSquare"                          : (-209.7147522, 436.5664062, 64, 60,  129),
    "BlowingBubbles"                    : (-207.0661316, 407.9269714, 64, 50,   97),
    "RaceHorses"                        : (-292.0694885, 508.1660614, 32, 30,   65),
    "TVD-01"                            : (-34.85756683, 74.62564850, 64, 50, 3000),
    "TVD-02"                            : (-34.49460220, 70.58546066, 64, 50,  636),
    "TVD-03"                            : (-38.38419342, 79.21981049, 64, 50, 2334),
    "2"                                 : (-14.86597729, 37.09463787, 32, 30, 4819),
    "13"                                : (-15.81821537, 37.90577221, 32, 30, 1416),
    "16"                                : (-14.20028019, 37.04280567, 32, 30,  700),
    "17"                                : (-11.13194656, 31.54760360, 32, 30,  966),
    "18"                                : (-14.42974854, 36.09946061, 32, 30, 1614),
}

SFU_CLASSWISE = {
    "classAB" : [
        "Traffic",
        "Kimono",
        "ParkScene",
        "Cactus", 
        "BasketballDrive",
        "BQTerrace",
    ],
    "classC":  [
        "BasketballDrill",
        "BQMall", 
        "PartyScene",
        "RaceHorsesC",
    ],
    "classD": [
        "BasketballPass",
        "BQSquare",
        "BlowingBubbles",
        "RaceHorses",
    ]
}

TVD_CLASSWISE = {
    "overall" : [
        "TVD-01",
        "TVD-02",
        "TVD-03",
    ],
}

HIEVE_CLASSWISE = {
    "1080p" : [
        "13",
        "16",
    ],
    "720p" : [
        "2",
        "17",
        "18",
    ],
}