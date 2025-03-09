export function addNumberInput(node, inputName, startValue, updateFunc, settings = { min: 16, max: 16384, step: 80, precision: 0, round: 1 }) {
   node.addWidget("number", inputName, startValue, updateFunc, settings);
}
