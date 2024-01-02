using AggressionScorerModel;

var builder = WebApplication.CreateBuilder(args);
// Add services to the container.



builder.Services.AddCors(options =>
{
    options.AddPolicy("MyPolicy", builder =>
    {
        builder.WithOrigins("http://localhost:3000")
               .AllowAnyMethod()
               .AllowAnyHeader();
    });
});
builder.Services.AddControllers();
builder.Services.AddAggressionScorePredictEnginPool();
builder.Services.AddSingleton<AggressionScore>();
var app = builder.Build();

// Configure the HTTP request pipeline.

app.UseCors("MyPolicy");
app.UseAuthorization();

app.MapControllers();

app.Run();
