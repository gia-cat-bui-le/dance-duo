using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// ReSharper disable MemberCanBePrivate.Global

namespace SMPLModel
{
    public class Bones : MonoBehaviour
    {
        public static readonly Dictionary<string, int> BoneIndex =
            new Dictionary<string, int>()
            {
                { "Pelvis", 0 },
                { "L_Hip", 1 },
                { "R_Hip", 2 },
                { "Spine1", 3 },
                { "L_Knee", 4 },
                { "R_Knee", 5 },
                { "Spine2", 6 },
                { "L_Ankle", 7 },
                { "R_Ankle", 8 },
                { "Spine3", 9 },
                { "L_Foot", 10 },
                { "R_Foot", 11 },
                { "Neck", 12 },
                { "L_Collar", 13 },
                { "R_Collar", 14 },
                { "Head", 15 },
                { "L_Shoulder", 16 },
                { "R_Shoulder", 17 },
                { "L_Elbow", 18 },
                { "R_Elbow", 19 },
                { "L_Wrist", 20 },
                { "R_Wrist", 21 },
                { "L_Hand", 22 },
                { "R_Hand", 23 }
            };

        public static string[] BoneNames =
        {
            "Pelvis",
            "L_Hip", "R_Hip", "Spine1",
            "L_Knee", "R_Knee", "Spine2",
            "L_Ankle", "R_Ankle", "Spine3",
            "L_Foot", "R_Foot", "Neck",
            "L_Collar", "R_Collar", "Head",
            "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow",
            "L_Wrist", "R_Wrist",
            "L_Hand", "R_Hand"
        };

        public static string[] BoneParent =
        {
            "Root",
            "Pelvis", "Pelvis", "Pelvis",
            "L_Hip", "R_Hip", "Spine1",
            "L_Knee", "R_Knee", "Spine2",
            "L_Ankle", "R_Ankle", "Spine3",
            "Spine3", "Spine3", "Neck",
            "L_Collar", "R_Collar",
            "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow",
            "L_Wrist", "R_Wrist"
        };

        public static string GetBoneRelativePath(string boneName)
        {
            var path = "/" + boneName;

            while (boneName != "Root")
            {
                path = $"/{BoneParent[BoneIndex[boneName]]}{path}";
                boneName = BoneParent[BoneIndex[boneName]];
            }

            path = path.TrimStart('/');

            return path;
        }

        public List<GameObject> bonesObjects;
        public List<Quaternion> initialRotations;

        public Quaternion GetBoneInitialRotation(int boneIndex)
        {
            return bonesObjects[boneIndex].transform.rotation;
        }

        private void Start()
        {
            initialRotations ??= new List<Quaternion>();

            foreach (var boneIndex in BoneIndex.Select(boneNameAndIndex => boneNameAndIndex.Value))
            {
                initialRotations.Add(GetBoneInitialRotation(boneIndex));
            }
        }
    }
}