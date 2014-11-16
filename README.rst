===========
poultry.box
===========

An example setup of poultry. To collect the tweets:

1. Provide oauth credentials in the poultry.cfg
2. Run::
    poultry -c poultry.cfg -s twitter://filter filter

Timeline
========

`timeline.dat` is a ``dot`` script that shows tweet bandwidth over a period of
time::

    poultry -s london timeline > timeline.txt
    gnuplot timeline.dat
    # open timeline.pdf
