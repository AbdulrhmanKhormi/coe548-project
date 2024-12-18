import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as rds from "aws-cdk-lib/aws-rds";
import * as efs from "aws-cdk-lib/aws-efs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as alb from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";

export class CdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, "coe548-vpc", {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: "private",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      ],
    });

    const secret = new secretsmanager.Secret(this, "db-secret", {
      secretName: "db-secret",
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "tariner" }),
        generateStringKey: "password",
        passwordLength: 16,
        excludePunctuation: true,
      },
    });

    const pg = new rds.DatabaseCluster(this, "trainingDB", {
      engine: rds.DatabaseClusterEngine.auroraPostgres({
        version: rds.AuroraPostgresEngineVersion.VER_16_4,
      }),
      credentials: rds.Credentials.fromSecret(secret),
      defaultDatabaseName: "train",
      vpc: vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
      port: 5432,
      clusterIdentifier: "trainingDB",
      writer: rds.ClusterInstance.provisioned("AuroraWriter", {
        instanceType: ec2.InstanceType.of(
          ec2.InstanceClass.BURSTABLE3,
          ec2.InstanceSize.LARGE,
        ),
      }),
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const fileSystem = new efs.FileSystem(this, "filesystem", {
      fileSystemName: "coe548-filesystem",
      vpc: vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const sharedMount = fileSystem.addAccessPoint(`sharedMount-AP`, {
      path: "/mnt/shared",
      posixUser: {
        uid: "1000",
        gid: "1000",
      },
      createAcl: {
        ownerGid: "1000",
        ownerUid: "1000",
        permissions: "755",
      },
    });

    const bucket = new s3.Bucket(this, "dataBucket", {
      bucketName: "coe548-train-data",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(
      this,
      "coe548-taskDefinition",
      {
        cpu: 256,
        memoryLimitMiB: 512,
        volumes: [
          {
            name: sharedMount.accessPointId,
            efsVolumeConfiguration: {
              fileSystemId: fileSystem.fileSystemId,
              transitEncryption: "ENABLED",
              authorizationConfig: {
                accessPointId: sharedMount.accessPointId,
                iam: "ENABLED",
              },
            },
          },
        ],
      },
    );

    const container = taskDefinition.addContainer("infernce", {
      image: ecs.ContainerImage.fromRegistry("akhormi/coe548:latest"),
      containerName: "infernce-container",
      environment: {
        BUCKET_NAME: bucket.bucketName,
        DB_HOST: pg.clusterEndpoint.hostname,
        DB_PORT: String(pg.clusterEndpoint.port),
        DB_NAME: "train",
        DB_USER: secret.secretValueFromJson("username").unsafeUnwrap(),
        DB_PASSWORD: secret.secretValueFromJson("password").unsafeUnwrap(),
      },
      portMappings: [{ containerPort: 8000 }],
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "infernce" }),
    });

    container.addMountPoints({
      sourceVolume: sharedMount.accessPointId,
      containerPath: "/shared",
      readOnly: false,
    });

    const cluster = new ecs.Cluster(this, "coe548-cluster", {
      vpc: vpc,
      clusterName: "coe548-cluster",
    });

    const service = new ecs.FargateService(this, "infernce-service", {
      cluster: cluster,
      taskDefinition: taskDefinition,
      serviceName: "infernce",
      desiredCount: 1,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
    });

    pg.connections.allowDefaultPortFrom(service);
    fileSystem.connections.allowDefaultPortFrom(service);
    fileSystem.grantReadWrite(service.taskDefinition.taskRole);
    bucket.grantReadWrite(service.taskDefinition.taskRole);

    const lb = new alb.ApplicationLoadBalancer(this, "coe548-alb", {
      vpc: vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
      },
      internetFacing: true,
      loadBalancerName: "coe548-alb",
    });

    const httpListener = lb.addListener("alb-listener-80", {
      port: 80,
      open: true,
    });

    httpListener.addTargets("infernce-target-group", {
      port: 8000,
      targets: [
        service.loadBalancerTarget({
          containerName: "infernce-container",
          containerPort: 8000,
        }),
      ],
    });

    new cdk.CfnOutput(this, "LoadBalancerDNS", {
      value: lb.loadBalancerDnsName,
      description: "DNS name for the load balancer",
    });
  }
}
