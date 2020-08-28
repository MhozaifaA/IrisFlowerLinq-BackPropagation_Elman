using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetworksAleppoth4
{
    //Dataset Order 	Sepal length 	Sepal width 	Petal length 	Petal width 	Species
    class Program
    {
        const  string path= @"../../../dataset/IrisDataset.txt";
        static void Main(string[] args)
        {
            Random random = new Random();
            const int InputNumber = 4;
            const float alpha = 0.1f;
            int epoch = 1000;

            HashSet<IrisFlower> IrisFlowers = ConvertFileToIrisFlower(path);

            // only one hidden layer
            List<OutValueTita> OutHiddenLayer = Enumerable.Range(0, InputNumber).  // new List<OutValueTita>(InputNumber)
                Select(o => new OutValueTita { Value = 0, Tita = random.NextDouble() }).ToList();

            List<WeightIrisFlower> WeightsInput = Enumerable.Range(0, InputNumber).//new List<WeightIrisFlower>(InputNumber).
                Select(f => new WeightIrisFlower()
                {
                    SepalLength = random.NextDouble(),
                    SepalWidth = random.NextDouble(),
                    PetalLength = random.NextDouble(),
                    PetalWidth = random.NextDouble(),
                }).ToList();

            // only  one output
            (double value, double tita) Output = (0, random.NextDouble());
            List<double> WeightsOutput = Enumerable.Range(0, InputNumber)  //  new List<double>(InputNumber);
                .Select(w => random.NextDouble()).ToList();
            

            Print("Start BackPropagation");

            BackPropagation(alpha, epoch, IrisFlowers, OutHiddenLayer, WeightsInput, Output, WeightsOutput);

            Print("Start Elman");

            Elman(alpha, epoch, IrisFlowers, OutHiddenLayer, WeightsInput, Output, WeightsOutput);

        }

        private static int BackPropagation(float alpha, int epoch, HashSet<IrisFlower> IrisFlowers, List<OutValueTita> OutHiddenLayer, List<WeightIrisFlower> WeightsInput, (double value, double tita) Output, List<double> WeightsOutput)
        {
            while (--epoch >= 0)
            {

                Print($"{nameof(epoch)} : {epoch}", ConsoleColor.Red);

                foreach (var IrisFlower in IrisFlowers)
                {

                    Print($"{nameof(IrisFlower)} Input data {IrisFlower.Id}:  " +
                        $"{IrisFlower.PetalLength} , {IrisFlower.PetalWidth} , {IrisFlower.SepalLength} , {IrisFlower.SepalWidth}",
                        ConsoleColor.DarkYellow);
                    Print($"{nameof(IrisFlower)} Output data {IrisFlower.Id}: {IrisFlower.Iris}", ConsoleColor.DarkYellow);


                    Print("calcaulate hidden values");
                    // calcaulate hidden values
                    foreach (var hiddenZipWeight in OutHiddenLayer.Zip(WeightsInput, (h, w) => new { hidden = h, Weight = w }))
                    {

                        hiddenZipWeight.hidden.Value = Sigmoid(IrisFlower.SepalLength * hiddenZipWeight.Weight.SepalLength +
                              IrisFlower.SepalWidth * hiddenZipWeight.Weight.SepalWidth +
                               IrisFlower.PetalLength * hiddenZipWeight.Weight.PetalLength +
                               IrisFlower.PetalWidth * hiddenZipWeight.Weight.PetalWidth - hiddenZipWeight.hidden.Tita);

                        Print($"Hidden value layer: {hiddenZipWeight.hidden.Value}", ConsoleColor.Cyan);
                    }


                    Print("calcaulate output value");
                    // calcaulate output value
                    Output.value = Sigmoid(OutHiddenLayer.Zip(WeightsOutput, (h, w) => new { hidden = h, Weight = w }).Sum(x => x.hidden.Value * x.Weight) - Output.tita);

                    Print($"output value layer: {Output.value}", ConsoleColor.Cyan);


                    double error = IrisFlower.Iris - Output.value;

                    Print($"error : {error}", ConsoleColor.Red);

                    //start weight fix


                    double delteOutput = Output.value * (1 - Output.value) * error;

                    Print($"delteOutput : {delteOutput}", ConsoleColor.Red);


                    /// order soo important


                    Print($"fix hidden Weights");
                    // fix hidden Weights
                    for (int i = 0; i < WeightsInput.Count; i++)
                    {
                        double deltaInput = (OutHiddenLayer[i].Value * (1 - OutHiddenLayer[i].Value) * delteOutput * WeightsOutput[i]);

                        Print($"deltaInput : {delteOutput}", ConsoleColor.Red);


                        WeightsInput[i].SepalLength += (alpha * IrisFlower.SepalLength * deltaInput);
                        WeightsInput[i].SepalWidth += (alpha * IrisFlower.SepalWidth * deltaInput);
                        WeightsInput[i].PetalLength += (alpha * IrisFlower.PetalLength * deltaInput);
                        WeightsInput[i].PetalWidth += (alpha * IrisFlower.PetalWidth * deltaInput);

                        Print($"new Weights for input {i} : \n {WeightsInput[i].PetalLength} \n {WeightsInput[i].PetalWidth}" +
                            $"\n {WeightsInput[i].SepalLength} \n {WeightsInput[i].SepalWidth}", ConsoleColor.Green);

                        // fix hidden tita
                        OutHiddenLayer[i].Tita += -alpha * deltaInput;
                        Print($"new tita hidden for neuron {i} : {OutHiddenLayer[i].Tita}", ConsoleColor.Green);

                    }



                    // fix output  tita
                    Output.tita += -alpha * delteOutput;
                    Print($"new tita  output : {Output.tita}", ConsoleColor.Green);

                    // fix output Weights
                    for (int i = 0; i < WeightsOutput.Count; i++)
                    {
                        WeightsOutput[i] += (alpha * delteOutput * OutHiddenLayer[i].Value);
                        Print($"new Weights for hidden {i} : \n {  WeightsOutput[i]}", ConsoleColor.Green);
                    }



                    Print($"Done fix Weights \n --------------------------------------------", ConsoleColor.Blue);
                    Print($"next data input");

                }


                Print($"Done round : {epoch}");

            }

            return epoch;
        }



        private static int Elman(float alpha, int epoch, HashSet<IrisFlower> IrisFlowers, List<OutValueTita> OutHiddenLayer, List<WeightIrisFlower> WeightsInput, (double value, double tita) Output, List<double> WeightsOutput)
        {
            Random random = new Random();
            const int InputNumber = 4;

            ///For elman
            // only one context layer
            List<OutValueTita> OutcontextLayer = Enumerable.Range(0, InputNumber).  // new List<OutValueTita>(InputNumber)
                Select(o => new OutValueTita { Value = -1, Tita = random.NextDouble() }).ToList();

            double WeightContext = random.NextDouble();
            List<WeightIrisFlower> WeightsContext = Enumerable.Range(0, InputNumber).//new List<WeightIrisFlower>(InputNumber).
            Select(f => new WeightIrisFlower()
            {
                SepalLength = WeightContext,
                SepalWidth = WeightContext,
                PetalLength = WeightContext,
                PetalWidth = WeightContext,
            }).ToList();


            while (--epoch >= 0)
            {

                Print($"{nameof(epoch)} : {epoch}", ConsoleColor.Red);

                foreach (var IrisFlower in IrisFlowers)
                {

                    Print($"{nameof(IrisFlower)} Input data {IrisFlower.Id}:  " +
                        $"{IrisFlower.PetalLength} , {IrisFlower.PetalWidth} , {IrisFlower.SepalLength} , {IrisFlower.SepalWidth}",
                        ConsoleColor.DarkYellow);
                    Print($"{nameof(IrisFlower)} Output data {IrisFlower.Id}: {IrisFlower.Iris}", ConsoleColor.DarkYellow);


                    Print("calcaulate hidden values");
                    // calcaulate hidden values
                    foreach (var hiddenZipWeight in OutHiddenLayer.Zip(OutcontextLayer,WeightsInput, WeightsContext,(oc,h, w,c) => new {contextvalue= oc, hidden = h, Weight = w , Context=c }))
                    {

                        double input = IrisFlower.SepalLength * hiddenZipWeight.Weight.SepalLength +
                              IrisFlower.SepalWidth * hiddenZipWeight.Weight.SepalWidth +
                               IrisFlower.PetalLength * hiddenZipWeight.Weight.PetalLength +
                               IrisFlower.PetalWidth * hiddenZipWeight.Weight.PetalWidth - hiddenZipWeight.hidden.Tita;


                        double context = OutcontextLayer[0].Value * hiddenZipWeight.Context.SepalLength +
                            OutcontextLayer[1].Value * hiddenZipWeight.Context.SepalWidth +
                             OutcontextLayer[2].Value * hiddenZipWeight.Context.PetalLength +
                              OutcontextLayer[3].Value * hiddenZipWeight.Context.PetalWidth - hiddenZipWeight.contextvalue.Tita;

                        hiddenZipWeight.hidden.Value = Sigmoid(input+context);

                        // copy value from hidden to context
                        hiddenZipWeight.contextvalue.Value = hiddenZipWeight.hidden.Value;

                        Print($"Hidden value layer: {hiddenZipWeight.hidden.Value}", ConsoleColor.Cyan);
                    }


                    Print("calcaulate output value");
                    // calcaulate output value
                    Output.value = Sigmoid(OutHiddenLayer.Zip(WeightsOutput,(h, w) => new { hidden = h, Weight = w }).Sum(x => x.hidden.Value * x.Weight) - Output.tita);

                    Print($"output value layer: {Output.value}", ConsoleColor.Cyan);


                    double error = IrisFlower.Iris - Output.value;

                    Print($"error : {error}", ConsoleColor.Red);

                    //start weight fix


                    double delteOutput = Output.value * (1 - Output.value) * error;

                    Print($"delteOutput : {delteOutput}", ConsoleColor.Red);


                    /// order soo important


                    Print($"fix hidden Weights");
                    // fix hidden Weights
                    for (int i = 0; i < WeightsInput.Count; i++)
                    {
                        double deltaInput = (OutHiddenLayer[i].Value * (1 - OutHiddenLayer[i].Value) * delteOutput * WeightsOutput[i]);

                        Print($"deltaInput : {delteOutput}", ConsoleColor.Red);


                        WeightsInput[i].PetalLength += (alpha * IrisFlower.PetalLength * deltaInput);
                        WeightsInput[i].PetalWidth += (alpha * IrisFlower.PetalWidth * deltaInput);
                        WeightsInput[i].SepalLength += (alpha * IrisFlower.SepalLength * deltaInput);
                        WeightsInput[i].SepalWidth += (alpha * IrisFlower.SepalWidth * deltaInput);

                        Print($"new Weights for input {i} : \n {WeightsInput[i].PetalLength} \n {WeightsInput[i].PetalWidth}" +
                            $"\n {WeightsInput[i].SepalLength} \n {WeightsInput[i].SepalWidth}", ConsoleColor.Green);

                        // fix hidden tita
                        OutHiddenLayer[i].Tita += -alpha * deltaInput;
                        Print($"new tita hidden for neuron {i} : {OutHiddenLayer[i].Tita}", ConsoleColor.Green);

                    }



                    // fix output  tita
                    Output.tita += -alpha * delteOutput;
                    Print($"new tita  output : {Output.tita}", ConsoleColor.Green);

                    // fix output Weights
                    for (int i = 0; i < WeightsOutput.Count; i++)
                    {
                        WeightsOutput[i] += (alpha * delteOutput * OutHiddenLayer[i].Value);
                        Print($"new Weights for hidden {i} : \n {  WeightsOutput[i]}", ConsoleColor.Green);
                    }



                    Print($"Done fix Weights \n --------------------------------------------", ConsoleColor.Blue);
                    Print($"next data input");

                }


                Print($"Done round : {epoch}");

            }

            return epoch;
        }


        public static HashSet<IrisFlower> ConvertFileToIrisFlower(string fileName)
        {
            List<string> FlowerName = new List<string>();
            string[] DataSetLine = File.ReadAllLines(fileName);
            HashSet<IrisFlower> IrisFlowers = new HashSet<IrisFlower>(DataSetLine.Length + 1);
            foreach (var item in DataSetLine)
            {
                var SplitOne = item.Split(" ").Select(s => s.Trim()).ToArray();

                int Output = FlowerName.FindIndex(x => x == SplitOne[5]);
                if (Output == -1)
                {
                    FlowerName.Add(SplitOne[5]);
                    Output = FlowerName.Count;
                }
                else
                {
                    Output = Output + 1;
                }

                IrisFlowers.Add(new IrisFlower()
                {

                    Id = Convert.ToInt32(SplitOne[0]),
                    SepalLength = Convert.ToDouble(SplitOne[1]),
                    SepalWidth = Convert.ToDouble(SplitOne[2]),
                    PetalLength = Convert.ToDouble(SplitOne[3]),
                    PetalWidth = Convert.ToDouble(SplitOne[4]),
                    Iris = Output
                });

            }
            return IrisFlowers;
        }


        public static double Sigmoid(double x, int multi=3)
        {
            return (1 / (1 + Math.Exp(-x)));
        }

        public static double Tansig(double x)
        {
            return Math.Exp(-x) /Math.Pow((1 + Math.Exp(-x)),2);
        }
      
        public static double Purlin(double x)
        {
            return x==0?0:(x>1?1:-1);
        }

        public static void Print(string text, ConsoleColor color = ConsoleColor.White)
        {
        
                Console.ForegroundColor = color;
                Console.WriteLine(text);
                Console.ResetColor();
        }

    }



    public class IrisFlower
    {
        public int Id { get; set; }
        public double SepalLength { get; set; } // input
        public double SepalWidth { get; set; } // input
        public double PetalLength { get; set; } // input
        public double PetalWidth { get; set; } // input
        public int Iris { get; set; } // output
    }


    public class WeightIrisFlower
    {
        public double SepalLength { get; set; }
        public double SepalWidth { get; set; } 
        public double PetalLength { get; set; }
        public double PetalWidth { get; set; } 
    }


    public class OutValueTita
    {
        public double Value { get; set; }
        public double Tita { get; set; }
    }


    public static class ExtensionsZip
    {
        public static IEnumerable<TResult> Zip<T1, T2, T3, T4, TResult>(
            this IEnumerable<T1> source,
            IEnumerable<T2> second,
            IEnumerable<T3> third,
             IEnumerable<T4> forth,
            Func<T1, T2, T3,T4, TResult> func)
        {
            using (var e1 = source.GetEnumerator())
            using (var e2 = second.GetEnumerator())
            using (var e3 = third.GetEnumerator())
            using (var e4 = forth.GetEnumerator())
            {
                while (e1.MoveNext() && e2.MoveNext() && e3.MoveNext() && e4.MoveNext())
                    yield return func(e1.Current, e2.Current, e3.Current,e4.Current);
            }
        }
    }

}


