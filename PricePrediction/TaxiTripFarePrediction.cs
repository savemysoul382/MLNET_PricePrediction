using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace PricePrediction
{
    /// <summary>
    /// The TaxiTripFarePrediction class represents a single far prediction.
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")] public float FareAmount;
    }
}
