import { PLYLoader } from '/static/threejs/jsm/loaders/PLYLoader.js';
import { PCDLoader } from '/static/threejs/jsm/loaders/PCDLoader.js';

self.onmessage = function (e) {
  const { buffer, type } = e.data;
  try {
    let geometry;
    if (type === 'pcd') {
      const loader = new PCDLoader();
      geometry = loader.parse(buffer);
    } else {
      const loader = new PLYLoader();
      loader.setCustomPropertyNameMapping({ label: ['class'] });
      geometry = loader.parse(buffer);
    }

    const result = {
      position: geometry.attributes.position.array.buffer,
      color: geometry.attributes.color ? geometry.attributes.color.array.buffer : null,
      label: geometry.attributes.label ? geometry.attributes.label.array.buffer : null,
    };
    const transfer = [result.position];
    if (result.color) transfer.push(result.color);
    if (result.label) transfer.push(result.label);
    self.postMessage({ success: true, ...result }, transfer);
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};
