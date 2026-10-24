"""Portable saved predictor wrapper; transforms fitted on training data only."""
import numpy as np
class Surrogate:
 def __init__(self,model,xscale,yscale):self.model=model;self.xscale=xscale;self.yscale=yscale
 def predict(self,x):return self.yscale.inverse_transform(self.model.predict(self.xscale.transform(np.atleast_2d(x))))


def export_models(models, path):
 """Save fitted coefficients and scaling values as non-executable JSON."""
 import json
 records = {}
 for name, wrapper in models.items():
  row = {'input_mean': wrapper.xscale.mean_.tolist(), 'input_scale': wrapper.xscale.scale_.tolist(),
         'output_mean': wrapper.yscale.mean_.tolist(), 'output_scale': wrapper.yscale.scale_.tolist()}
  if name == 'mlp':
   row.update(kind='logistic_mlp', weights=[a.tolist() for a in wrapper.model.coefs_],
              biases=[a.tolist() for a in wrapper.model.intercepts_])
  elif name == 'quadratic_ridge':
   polynomial, ridge = wrapper.model.steps[0][1], wrapper.model.steps[1][1]
   row.update(kind='quadratic_ridge', powers=polynomial.powers_.tolist(),
              coefficients=ridge.coef_.tolist(), intercept=ridge.intercept_.tolist())
  else:
   raise ValueError('Unsupported predictor: ' + name)
  records[name] = row
 path.write_text(json.dumps({'schema': 1, 'models': records}, indent=2) + '\n')


class JsonPredictor:
 """Evaluate released fitted models without pickle or scikit-learn loading."""
 def __init__(self, record):
  self.record = record
 def predict(self, x):
  r = self.record
  a = (np.atleast_2d(x) - np.asarray(r['input_mean'])) / np.asarray(r['input_scale'])
  if a.shape[1] != 9 or not np.all(np.isfinite(a)):
   raise ValueError('Expected finite features: seven feeds, log initial ratio, initial glucose')
  if r['kind'] == 'logistic_mlp':
   from scipy.special import expit
   for i, (w, b) in enumerate(zip(r['weights'], r['biases'])):
    a = a @ np.asarray(w) + np.asarray(b)
    if i < len(r['weights']) - 1:
     a = expit(a)
  elif r['kind'] == 'quadratic_ridge':
   powers = np.asarray(r['powers'], dtype=int)
   a = np.prod(a[:, None, :] ** powers[None, :, :], axis=2)
   a = a @ np.asarray(r['coefficients']).T + np.asarray(r['intercept'])
  else:
   raise ValueError('Unknown model type')
  return a * np.asarray(r['output_scale']) + np.asarray(r['output_mean'])


def load_models(path):
 import json
 data = json.loads(path.read_text())
 if data['schema'] != 1:
  raise ValueError('Unsupported predictor schema')
 return {name: JsonPredictor(record) for name, record in data['models'].items()}
