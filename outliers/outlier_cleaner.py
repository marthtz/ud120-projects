#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    # Get error of predection to net worths
    err = predictions - net_worths

    # Create a new tuple list
    TList = []
    for Index, Item in enumerate(predictions):
        TList.append((ages[Index][0], net_worths[Index][0], abs(err[Index][0])))

    # Sort by error
    sortedList = sorted(TList, key=lambda x: x[2])

    # Determine data length and return only 90%
    dataLen = len(sortedList)
    cleaned_data = sortedList[:int(dataLen*0.9)]

    return cleaned_data

