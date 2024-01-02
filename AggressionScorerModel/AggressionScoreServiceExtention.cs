using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.ML;


namespace AggressionScorerModel
{
    public static class AggressionScoreServiceExtention
    {
        private static readonly string _modelPath =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model", "AggressionScorerRetranedModel.zip");
        public static void AddAggressionScorePredictEnginPool(this IServiceCollection services)
        {
            services.AddPredictionEnginePool<ModelInput, ModelOutPut>()
                .FromFile(_modelPath,true);
        }
    }
}
