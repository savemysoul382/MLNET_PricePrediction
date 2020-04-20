using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace PricePrediction
{
    /// <summary>
    /// The TaxiTrip class represents a single taxi trip.
    /// </summary>
    public class TaxiTrip
    {
        [LoadColumn(0)] public string VendorId;
        [LoadColumn(5)] public string RateCode;
        [LoadColumn(3)] public float PassengerCount;
        [LoadColumn(4)] public float TripDistance;
        [LoadColumn(9)] public string PaymentType;
        [LoadColumn(10)] public float FareAmount;
    }
}