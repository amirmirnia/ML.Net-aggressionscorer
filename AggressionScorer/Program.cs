using AggressionScorer;
using AggressionScorerModel;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

Console.WriteLine("Aggression score model builder start");
var mlcontext = new MLContext(0);

//load data
var creatInputFile = @"Data\preparedInput.tsv";
DataPreparer.CreatePreparedDataFile(creatInputFile, true);

IDataView trainDataView = mlcontext.Data.LoadFromTextFile<ModelInput>(
    path: creatInputFile,
    hasHeader: true,
    separatorChar: '\t',
    allowQuoting: true
    );
var InputDataSplit = mlcontext.Data.TrainTestSplit(trainDataView, .2, null, 0);


//build pipeline
var inputDataPrepare = mlcontext
    .Transforms
    .Text
    .FeaturizeText(inputColumnName: "comment", outputColumnName: "Features")
    .Append(mlcontext.Transforms.NormalizeMeanVariance("Features"))
    .AppendCacheCheckpoint(mlcontext)
    .Fit(InputDataSplit.TrainSet);

var trainer = mlcontext
    .BinaryClassification
    .Trainers
    .LbfgsLogisticRegression();

//var trainerPipeline = inputDataPrepare.Append(trainer);


//fit the model
Console.Write("Start Taining");
var transformdata = inputDataPrepare.Transform(InputDataSplit.TrainSet);
ITransformer model = trainer.Fit(transformdata);

//test data

EvaluateData(mlcontext, model, inputDataPrepare.Transform(InputDataSplit.TestSet));

//save the model
if (!Directory.Exists("Model"))
{
    Directory.CreateDirectory("Model");

}

var modelFile = @"Model\\AggressionScorerModel.zip";

mlcontext.Model.Save(model, trainDataView.Schema, modelFile);
Console.WriteLine("Done!");

var Datapreparepiplinefile = @"Model\\Datapreparepipline.zip";

mlcontext.Model.Save(inputDataPrepare, trainDataView.Schema, Datapreparepiplinefile);

var ReTModel = RetrainModel(modelFile, Datapreparepiplinefile);
var compliteretrainpipline = inputDataPrepare.Append(ReTModel);
Console.WriteLine("Save Retrain Model");
var Retrainmodelfile= @"Model\\AggressionScorerRetranedModel.zip";
mlcontext.Model.Save(compliteretrainpipline, trainDataView.Schema, Retrainmodelfile);
Console.WriteLine("Model save {0}", Retrainmodelfile);
EvaluateData(mlcontext, compliteretrainpipline, inputDataPrepare.Transform(InputDataSplit.TestSet));


ITransformer RetrainModel(string model, string inputDataPrepare)
{
    MLContext mlcontext = new MLContext(0);
    ITransformer pretrainmodel = mlcontext.Model.Load(model, out _);
    var oretraminModelParameter = ((ISingleFeaturePredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>)
        pretrainmodel).Model.SubModel;//استخراج هوش یاد گیری مودل

    var datafile = @"Data\preparedInput.tsv";
    DataPreparer.CreatePreparedDataFile(datafile, true);

    IDataView trainDataView = mlcontext.Data.LoadFromTextFile<ModelInput>(
        path: datafile,
        hasHeader: true,
        separatorChar: '\t',
        allowQuoting: true
        );


    ITransformer datapreparepipline = mlcontext.Model.Load(inputDataPrepare, out _);
    var newdata = datapreparepipline.Transform(trainDataView);

    var returnModel =
        mlcontext.BinaryClassification.Trainers.LbfgsLogisticRegression().Fit(newdata, oretraminModelParameter);

    return returnModel;

}

void EvaluateData(MLContext mlcontext,ITransformer TrainerData,IDataView TestData)
{
    Console.WriteLine();
    Console.WriteLine("--Evaluate BinaryClassification--");
    Console.WriteLine();

    var predictdata = TrainerData.Transform(TestData);
    var Metrix = mlcontext.BinaryClassification.Evaluate(predictdata);
    Console.WriteLine($"Accuracy:{ Metrix.Accuracy}");
    Console.WriteLine("Confusion Matrix");
    Console.WriteLine();
    Console.WriteLine(Metrix.ConfusionMatrix.GetFormattedConfusionTable());
}