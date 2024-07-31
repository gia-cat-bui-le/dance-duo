using UnityEngine;

namespace Playground
{
    public class Rotate : MonoBehaviour
    {
        public Vector3 pelvisRotation;
        public Vector3 lHipRotation;
        public Vector3 lKneeRotation;
        public bool doGlobal2Local;
        public bool doRotate;

        public GameObject pelvis;
        public GameObject lHip;
        public GameObject lKnee;

        // Start is called before the first frame update
        void Start()
        {
            pelvisRotation = new Vector3((float)86.7223, (float)-0.694554,
                (float)13.6565);
            lHipRotation = new Vector3((float)-10.396, (float)-0.241948, (float)14.6605);
            lKneeRotation = new Vector3((float)20.0542, (float)0.495125,
                (float)-1.80693);
            // if (doRotate)
            // {
            //     RotateVector();
            // }
            // RotateVector();
            if (doGlobal2Local)
            {
                Global2Local();
            }

            pelvis.transform.localRotation = Quaternion.Euler(pelvisRotation);
            lHip.transform.localRotation = Quaternion.Euler(lHipRotation);
            lKnee.transform.localRotation = Quaternion.Euler(lKneeRotation);
        }

        private void Global2Local()
        {
            var lHipQuaternion = Quaternion.Euler(lHipRotation);
            var pelvisQuaternion = Quaternion.Euler(pelvisRotation);
            var lKneeQuaternion = Quaternion.Euler(lKneeRotation);

            var lHipQuaternionLocal = Quaternion.Inverse(pelvisQuaternion) * lHipQuaternion;
            var lKneeQuaternionLocal = Quaternion.Inverse(lHipQuaternionLocal) * lKneeQuaternion;
            lHipRotation = lHipQuaternionLocal.eulerAngles;
            if (doGlobal2Local)
            {
                lKneeRotation = lKneeQuaternionLocal.eulerAngles;
            }
        }

        private void RotateVector()
        {
            var rotationVector = Quaternion.AngleAxis(90, Vector3.left);

            pelvisRotation = rotationVector * pelvisRotation;
            lHipRotation = rotationVector * lHipRotation;
        }

        // Update is called once per frame
        void Update()
        {
        }
    }
}