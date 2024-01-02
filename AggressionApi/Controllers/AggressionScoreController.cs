using AggressionScorerModel;
using Microsoft.AspNetCore.Cors;
using Microsoft.AspNetCore.Mvc;

namespace AggressionApi.Controllers
{
    [EnableCors("MyPolicy")]
    [ApiController]
    [Route("[controller]")]
    public class AggressionScoreController : Controller
    {
        private readonly AggressionScore _aggressionScore;
        public AggressionScoreController(AggressionScore aggressionScore)
        {
            _aggressionScore = aggressionScore;
        }

        [HttpPost]
        public JsonResult Post([FromBody] CommentWrapper input)
        {
            
            return new JsonResult(ScoreComment(input.Comment));
        }
        private AggressionPrediction ScoreComment(string comment)
        {
            var classification = _aggressionScore.predict(comment);
            return new AggressionPrediction()
            {
                IsAggressive = classification.Prediction,
                Probability = classification.Probability
            };
        }
        public class AggressionPrediction
        {
            public bool IsAggressive { get; set; }
            public float Probability { get; set; }
        }

        public class CommentWrapper
        {
            public string Comment { get; set; }
        }
    }
}
