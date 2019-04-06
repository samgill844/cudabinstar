import numpy as np 
from astropy import constants

# https://watermark.silverchair.com/stt2100.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAkQwggJABgkqhkiG9w0BBwagggIxMIICLQIBADCCAiYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMWCc6d7k8yr-WHlQGAgEQgIIB9-L18kNuqVAPGu9fIyhp9Ofc2GX84wW_8V6OgWzPpnFG5SptGetiyb1tQ-tvGIHAgtXiXDOot2dTVj6TnTOY2DUNWK1BTXIajsaXX2FepKzaG6wZvI0cHMftf_CscpHnPgqygz94PEjWC5UWQ17il7xKJxZYmhqIFldPOCz3vsluEUCgRyiVz35qaiO6SEgflTYwWqmJcs_Tg2L-3LgSioNcRYqDl8SrsJ2GfVBUHavkbt4almX4TciRp63PAc03tY5ciJGWw8wng7tmxsRKVLYEfh9RnNVpmk5LG8z6PJwLDavn8FfGR0j3Pmh5oa_40SivVtxrAKucuiZOK_7iy4m-yd3j6gLB1p_W3jzO_t62fX0XPpYdX2ehOHzwRdOQ6EjjoP3kX3uPxTer7bzEK5L7O4VQqEWNrCmTf7o7X48YM_mIx3dEQ4OjIbQtnPvvG2F52dKfrRg6243mcMISx-BW3klGOY2vcpPMiODTgSpoqSBG1HGTr3qC9JhIwX0KurAw_Dw3eMVcj9cCHxxfVFlI5S43WMoEj4a-aZc91puhtLZ80wglt-c_uG49Mq1uD1fgiXGyd5a7TaUlYbwst7cg66nAafSl5p4-YTTG-t-MCVqgA7MyTyQC76yvOH7P4qhF1TJsQgkoLU5Amiqz51LyOsi5uOby


def M1(period,radius_1,vsini, K1, e, incl) :
    period *= 86400 # days -> s
    vsini *= 1e3 # km/s -> m/s
    incl = np.pi*incl/180 # convert to radians
    K1 *= 1e3

    return (period/(2*np.pi*constants.G.value))*(1/radius_1)**2 * vsini**2 * (( (1/radius_1)*vsini - K1*np.sqrt(1-e**2)) / (np.sin(incl)**3) ) / constants.M_sun.value


def M2(period,radius_1,vsini, K1, e, incl) :
    period *= 86400 # days -> s
    vsini *= 1e3 # km/s -> m/s
    incl = np.pi*incl/180 # convert to radians
    K1 *= 1e3

    return (period/(2*np.pi*constants.G.value))*(1/radius_1)**2 * ((K1*vsini**2*np.sqrt(1-e**2))/(np.sin(incl)**3))/ constants.M_sun.value
    

def R1(period, vsini, incl) : 
    period *= 86400 # days -> s
    vsini *= 1e3 # km/s -> m/s
    incl = np.pi*incl/180 # convert to radians

    return (period/(2*np.pi))*vsini / np.sin(incl) / constants.R_sun.value


def all(period,radius_1,vsini, K1, e, incl, k):
    print(M1(period,radius_1,vsini, K1, e, incl))
    print(R1(period, vsini, incl))

    print(M2(period,radius_1,vsini, K1, e, incl))
    print(R1(period, vsini, incl)*k)
