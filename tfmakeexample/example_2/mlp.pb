
;
xPlaceholder*
shape:��������� *
dtype0
;
yPlaceholder*
shape:���������*
dtype0
K
truncated_normal/shapeConst*
valueB"       *
dtype0
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
valueB
 *���=*
dtype0
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
\
Variable
VariableV2*
shared_name *
dtype0*
	container *
shape
: 
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
6
ConstConst*
valueB*    *
dtype0
Z

Variable_1
VariableV2*
shared_name *
dtype0*
	container *
shape:

Variable_1/AssignAssign
Variable_1Const*
_class
loc:@Variable_1*
validate_shape(*
use_locking(*
T0
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
M
truncated_normal_1/shapeConst*
valueB"      *
dtype0
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
^

Variable_2
VariableV2*
shared_name *
dtype0*
	container *
shape
:
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
8
Const_1Const*
valueB*    *
dtype0
Z

Variable_3
VariableV2*
shared_name *
dtype0*
	container *
shape:
�
Variable_3/AssignAssign
Variable_3Const_1*
_class
loc:@Variable_3*
validate_shape(*
use_locking(*
T0
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
Q
MatMulMatMulxVariable/read*
transpose_a( *
transpose_b( *
T0
K
BiasAddBiasAddMatMulVariable_1/read*
T0*
data_formatNHWC

TanhTanhBiasAdd*
T0
X
MatMul_1MatMulTanhVariable_2/read*
transpose_a( *
transpose_b( *
T0
O
	BiasAdd_1BiasAddMatMul_1Variable_3/read*
T0*
data_formatNHWC
!
y_outTanh	BiasAdd_1*
T0

subSubyy_out*
T0

SquareSquaresub*
T0
<
Const_2Const*
valueB"       *
dtype0
B
costSumSquareConst_2*

Tidx0*
	keep_dims( *
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
V
!gradients/cost_grad/Reshape/shapeConst*
valueB"      *
dtype0
p
gradients/cost_grad/ReshapeReshapegradients/Fill!gradients/cost_grad/Reshape/shape*
T0*
Tshape0
C
gradients/cost_grad/ShapeShapeSquare*
out_type0*
T0
s
gradients/cost_grad/TileTilegradients/cost_grad/Reshapegradients/cost_grad/Shape*

Tmultiples0*
T0
c
gradients/Square_grad/ConstConst^gradients/cost_grad/Tile*
valueB
 *   @*
dtype0
K
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*
T0
`
gradients/Square_grad/Mul_1Mulgradients/cost_grad/Tilegradients/Square_grad/Mul*
T0
=
gradients/sub_grad/ShapeShapey*
T0*
out_type0
C
gradients/sub_grad/Shape_1Shapey_out*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0
�
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
�
gradients/sub_grad/Sum_1Sumgradients/Square_grad/Mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
@
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0
r
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
h
gradients/y_out_grad/TanhGradTanhGrady_out-gradients/sub_grad/tuple/control_dependency_1*
T0
r
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/y_out_grad/TanhGrad*
data_formatNHWC*
T0
x
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp%^gradients/BiasAdd_1_grad/BiasAddGrad^gradients/y_out_grad/TanhGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/y_out_grad/TanhGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/y_out_grad/TanhGrad
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
�
gradients/MatMul_1_grad/MatMulMatMul1gradients/BiasAdd_1_grad/tuple/control_dependencyVariable_2/read*
T0*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulTanh1gradients/BiasAdd_1_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
i
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0
o
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Tanh_grad/TanhGrad*
data_formatNHWC*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/BiasAdd_grad/BiasAddGrad^gradients/Tanh_grad/TanhGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Tanh_grad/TanhGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Tanh_grad/TanhGrad
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
�
gradients/MatMul_grad/MatMulMatMul/gradients/BiasAdd_grad/tuple/control_dependencyVariable/read*
T0*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulx/gradients/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
c
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0
t
beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@Variable*
dtype0*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
loc:@Variable*
validate_shape(*
use_locking(*
T0
O
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable
c
beta2_power/initial_valueConst*
dtype0*
valueB
 *w�?*
_class
loc:@Variable
t
beta2_power
VariableV2*
shared_name *
_class
loc:@Variable*
dtype0*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
O
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable
q
Variable/Adam/Initializer/zerosConst*
dtype0*
valueB *    *
_class
loc:@Variable
~
Variable/Adam
VariableV2*
	container *
shape
: *
shared_name *
_class
loc:@Variable*
dtype0
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
S
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable
s
!Variable/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@Variable*
dtype0
�
Variable/Adam_1
VariableV2*
_class
loc:@Variable*
dtype0*
	container *
shape
: *
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
W
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable
q
!Variable_1/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_1*
dtype0
~
Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
dtype0*
	container *
shape:*
shared_name 
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1
s
#Variable_1/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
dtype0*
	container *
shape:*
shared_name 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0
u
!Variable_2/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@Variable_2
�
Variable_2/Adam
VariableV2*
dtype0*
	container *
shape
:*
shared_name *
_class
loc:@Variable_2
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0
w
#Variable_2/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
dtype0*
	container *
shape
:
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2
q
!Variable_3/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_3*
dtype0
~
Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
dtype0*
	container *
shape:*
shared_name 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3
s
#Variable_3/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_3*
dtype0
�
Variable_3/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class
loc:@Variable_3
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3
@
train/learning_rateConst*
valueB
 *o�:*
dtype0
8
train/beta1Const*
valueB
 *fff?*
dtype0
8
train/beta2Const*
valueB
 *w�?*
dtype0
:
train/epsilonConst*
valueB
 *w�+2*
dtype0
�
train/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable*
use_nesterov( *
use_locking( *
T0
�
!train/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( 
�
!train/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
use_nesterov( *
use_locking( *
T0
�
!train/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( 
�
	train/mulMulbeta1_power/readtrain/beta1 ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam*
_class
loc:@Variable*
T0
}
train/AssignAssignbeta1_power	train/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(
�
train/mul_1Mulbeta2_power/readtrain/beta2 ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam*
T0*
_class
loc:@Variable
�
train/Assign_1Assignbeta2_powertrain/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(
�
trainNoOp^train/Assign^train/Assign_1 ^train/update_Variable/ApplyAdam"^train/update_Variable_1/ApplyAdam"^train/update_Variable_2/ApplyAdam"^train/update_Variable_3/ApplyAdam
�
init_all_vars_opNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^beta1_power/Assign^beta2_power/Assign"