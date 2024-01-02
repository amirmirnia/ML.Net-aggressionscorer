using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.ML;

namespace AggressionScorerModel
{
    public class AggressionScore
    {
        private readonly PredictionEnginePool<ModelInput, ModelOutPut> _predictionEnginePool;

        public AggressionScore(PredictionEnginePool<ModelInput, ModelOutPut> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }
        public ModelOutPut predict(string input) =>
            _predictionEnginePool.Predict(new ModelInput
            {
                comment = input
            });
    }
}
